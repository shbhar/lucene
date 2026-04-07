/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.codecs.turboquant;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.util.VectorUtil;

/**
 * SIMD-accelerated scorer for TurboQuant compressed vectors.
 *
 * <p>Uses Java Vector API (JDK 25 incubator) for:
 *
 * <ul>
 *   <li>8-bit: int8 dot product via B2S widening multiply + accumulate
 *   <li>4-bit: ByteVector.rearrange (NEON TBL / x86 PSHUFB) for centroid gather
 *   <li>1-bit: scalar int popcount (already fast, no SIMD needed)
 * </ul>
 *
 * @lucene.experimental
 */
public final class TurboQuantScorer {

  // TODO(SIMD portability): B128 assumes 128-bit SIMD (NEON TBL / SSE PSHUFB). On AVX2+,
  // VectorSpecies.ofPreferred(byte.class) would yield SPECIES_256 for 32-byte VPSHUFB.
  // Replace B128 with SPECIES_PREFERRED and dispatch on species.length() for cross-platform
  // performance. Requires adjusting loop strides and rearrange shuffle masks accordingly.
  private static final VectorSpecies<Byte> B128 = ByteVector.SPECIES_128;

  /** Below this norm, a vector is treated as zero. */
  static final float NORM_EPSILON = 1e-30f;

  private TurboQuantScorer() {}

  /**
   * Precomputed query state for fast scoring against TurboQuant-compressed document vectors.
   * Created once per query via {@link #prepareQuery(float[], float[], int)} and reused across
   * all document scorings in a search. Holds the FWHT-rotated query, int8 quantized forms for
   * SIMD dot products, centroid LUTs for bin-to-value mapping, and precomputed bit-plane
   * representations for 1-bit popcount scoring. Immutable after construction.
   */
  static final class QueryState {
    final float[] rotatedQuery;
    final float queryNorm;
    final int bits;
    final float[] centroidLUT;
    final byte[] queryInt8;
    final float int8InvScale;
    final byte[] centroidInt8;
    final byte[] queryInt8Deinterleaved;
    final byte[] queryBitsPacked;
    final float totalRotatedQuery;
    final float sumRotatedQueryPositive;
    final float meanRotatedQueryPositive;
    final float meanRotatedQueryNegative;
    final byte[] queryInt8Planes;
    final int[] queryPlanesInt;
    final float planeInvScale;
    final float planeBias;

    QueryState(
        float[] rotatedQuery,
        float queryNorm,
        int bits,
        float[] centroidLUT,
        byte[] queryInt8,
        float int8InvScale,
        byte[] centroidInt8,
        byte[] queryInt8Deinterleaved) {
      this.rotatedQuery = rotatedQuery;
      this.queryNorm = queryNorm;
      this.bits = bits;
      this.centroidLUT = centroidLUT;
      this.queryInt8 = queryInt8;
      this.int8InvScale = int8InvScale;
      this.centroidInt8 = centroidInt8;
      this.queryInt8Deinterleaved = queryInt8Deinterleaved;

      int dim = rotatedQuery.length;
      int packedLen = (dim + 7) >>> 3;
      this.queryBitsPacked = new byte[packedLen];
      float tq = 0, sp = 0;
      int numPos = 0;
      for (int i = 0; i < dim; i++) {
        tq += rotatedQuery[i];
        if (rotatedQuery[i] >= 0) {
          sp += rotatedQuery[i];
          numPos++;
          queryBitsPacked[i >>> 3] |= (byte) (1 << (i & 7));
        }
      }
      this.totalRotatedQuery = tq;
      this.sumRotatedQueryPositive = sp;
      this.meanRotatedQueryPositive = numPos > 0 ? sp / numPos : 0;
      this.meanRotatedQueryNegative = (dim - numPos) > 0 ? (tq - sp) / (dim - numPos) : 0;

      // 4-bit-plane packed query for fast 1-bit scoring
      this.queryInt8Planes = new byte[4 * packedLen];
      float absMax = 0;
      for (float v : rotatedQuery) absMax = Math.max(absMax, Math.abs(v));
      float rScale4 = absMax > 0 ? 7.5f / absMax : 0f;
      this.planeInvScale = absMax > 0 ? absMax / 7.5f : 0f;
      this.planeBias = -absMax;
      for (int i = 0; i < dim; i++) {
        int qVal = Math.max(0, Math.min(15, Math.round((rotatedQuery[i] + absMax) * rScale4)));
        int byteIdx = i >>> 3, bitIdx = i & 7;
        for (int p = 0; p < 4; p++) {
          if (((qVal >>> p) & 1) != 0) {
            queryInt8Planes[p * packedLen + byteIdx] |= (byte) (1 << bitIdx);
          }
        }
      }
      int intsPerStripe = packedLen >>> 2;
      this.queryPlanesInt = new int[4 * intsPerStripe];
      for (int p = 0; p < 4; p++) {
        int pOff = p * packedLen;
        int iOff = p * intsPerStripe;
        for (int ii = 0; ii < intsPerStripe; ii++) {
          int bOff = pOff + (ii << 2);
          queryPlanesInt[iOff + ii] =
              (queryInt8Planes[bOff] & 0xFF)
                  | ((queryInt8Planes[bOff + 1] & 0xFF) << 8)
                  | ((queryInt8Planes[bOff + 2] & 0xFF) << 16)
                  | ((queryInt8Planes[bOff + 3] & 0xFF) << 24);
        }
      }
    }
  }

  /**
   * Shared rotation logic: normalize → sign-flip → FWHT → build centroid LUT.
   * Used by both {@link #prepareQuery} (full scoring path) and the float-precision
   * rescorer to guarantee identical rotation and LUT construction.
   *
   * @param query input query vector (not modified)
   * @param signs random sign flip array from encoder seed
   * @param bits quantization bit depth (1, 2, 4, or 8)
   * @return {rotated, {queryNorm}, lut} — rotated unit vector, norm scalar, centroid LUT
   */
  static float[][] rotateAndBuildLUT(float[] query, float[] signs, int bits) {
    final int dim = query.length;
    float norm = 0;
    for (float v : query) norm += v * v;
    norm = (float) Math.sqrt(norm);

    float[] rotated = new float[dim];
    if (norm > NORM_EPSILON) {
      float invNorm = 1.0f / norm;
      for (int i = 0; i < dim; i++) rotated[i] = query[i] * invNorm;
    }
    FWHT.applySignFlips(rotated, signs);
    FWHT.transform(rotated);

    float sigma = 1.0f / (float) Math.sqrt(dim);
    int levels = PolarQuant.levels(bits);
    float[] centroids = PolarQuant.centroids(bits);
    float[] lut = new float[levels];
    for (int j = 0; j < levels; j++) lut[j] = centroids[j] * sigma;

    return new float[][] {rotated, {norm}, lut};
  }

  /**
   * Prepare query state for scoring at the specified bit depth.
   *
   * @param query input query vector (not modified)
   * @param signs random sign flip array from encoder seed
   * @param bits quantization bit depth (1, 2, 4, or 8)
   * @return precomputed query state for repeated scoring
   */
  static QueryState prepareQuery(float[] query, float[] signs, int bits) {
    final int dim = query.length;
    float[][] rl = rotateAndBuildLUT(query, signs, bits);
    float[] rotated = rl[0];
    float norm = rl[1][0];
    float[] lut = rl[2];

    // Int8 quantization of rotated query
    float qMax = 0;
    for (float v : rotated) qMax = Math.max(qMax, Math.abs(v));
    float qScale = (qMax > 0) ? 127f / qMax : 1f;
    byte[] queryInt8 = new byte[dim];
    for (int i = 0; i < dim; i++) queryInt8[i] = (byte) Math.round(rotated[i] * qScale);

    // Int8 centroid LUT (padded to 16 for 4-bit selectFrom)
    float cMax = 0;
    for (float v : lut) cMax = Math.max(cMax, Math.abs(v));
    float cScale = (cMax > 0) ? 127f / cMax : 1f;
    int centroidInt8Len = (bits == 4) ? 16 : lut.length;
    byte[] centroidInt8 = new byte[centroidInt8Len];
    for (int j = 0; j < lut.length; j++) centroidInt8[j] = (byte) Math.round(lut[j] * cScale);

    float int8InvScale = 1.0f / (qScale * cScale);

    // Deinterleave: [all evens | all odds]
    int half = dim / 2;
    byte[] deinterleaved = new byte[dim];
    for (int i = 0; i < half; i++) {
      deinterleaved[i] = queryInt8[2 * i];
      deinterleaved[half + i] = queryInt8[2 * i + 1];
    }

    return new QueryState(
        rotated,
        norm,
        bits,
        lut,
        queryInt8,
        int8InvScale,
        centroidInt8,
        deinterleaved);
  }

  /**
   * Primary scoring path. Uses int8 SIMD for 4-bit, bulk unpack for 1/2-bit, float for 8-bit.
   *
   * @param query precomputed query state from {@link #prepareQuery}
   * @param doc quantized document vector
   * @return approximate dot product score (queryNorm × docNorm × polarCos)
   */
  public static float score(QueryState query, TurboQuantVector doc) {
    final float docNorm = doc.getNorm();
    if (docNorm < 1e-30f || query.queryNorm < 1e-30f) return 0f;
    final int dim = doc.dim();
    if (query.bits <= 2) {
      float polarCos = scorePolarBulk(query, doc, dim, query.bits);
      return query.queryNorm * docNorm * polarCos;
    }
    if (query.bits == 8) {
      // 8-bit packed: 1 bin per byte, 256 levels — use float LUT path
      float polarCos = scorePolarFloat(query.rotatedQuery, query.centroidLUT, doc, dim);
      return query.queryNorm * docNorm * polarCos;
    }
    // 4-bit: direct float LUT accumulation — no intermediate array
    float polarCos = scorePolar4BitFloatDirect(
        query.rotatedQuery, query.centroidLUT,
        doc.segment(), TurboQuantVector.PACKED_BINS_OFFSET, dim / 2, dim);
    return query.queryNorm * docNorm * polarCos;
  }

  /** Float-precision scoring path (higher accuracy, useful for low-bit rescoring). */
  public static float scoreFloat(QueryState query, TurboQuantVector doc) {
    final float docNorm = doc.getNorm();
    if (docNorm < 1e-30f || query.queryNorm < 1e-30f) return 0f;
    float polarCos = scorePolarFloat(query.rotatedQuery, query.centroidLUT, doc, doc.dim());
    return query.queryNorm * docNorm * polarCos;
  }

  // --- Thread-local buffers ---

  // Thread-local scratch buffers for bin expansion. Avoids per-call allocation in the scorer
  // hot path. Sized lazily to match the vector dimension. In application server environments,
  // these may persist on pooled threads — acceptable tradeoff vs per-call GC pressure during
  // HNSW search where the scorer is invoked thousands of times per query.
  // TODO(turboquant): consider passing buffers from the scorer supplier to avoid ThreadLocal entirely.
  // TODO(turboquant): These ThreadLocal buffers grow to max dimension seen but never shrink.
  // Consider passing buffers from the scorer supplier or using a size-capped pool.
  private static final ThreadLocal<byte[]> BIN_BUF = ThreadLocal.withInitial(() -> new byte[4096]);

  /**
   * Int8 polar scorer for packed 4-bit bins. Query is pre-deinterleaved: {@code
   * [q[0],q[2],...,q[dim-2], q[1],q[3],...,q[dim-1]]}. Each packed byte contains 2 bins (lo
   * nibble=even, hi nibble=odd).
   */
  private static int scorePolarInt8(
      byte[] queryInt8, byte[] centroidInt8, TurboQuantVector doc, int dim) {
    int packedLen = TurboQuantVector.packedBinsBytes(dim, 4);
    byte[] packed = BIN_BUF.get();
    if (packed.length < packedLen) {
      packed = new byte[packedLen];
      BIN_BUF.set(packed);
    }
    MemorySegment.copy(
        doc.segment(),
        TurboQuantVector.PACKED_BINS_OFFSET,
        MemorySegment.ofArray(packed),
        0,
        packedLen);

    final int half = dim / 2;
    int sum = 0;
    for (int pi = 0; pi < packedLen; pi++) {
      int p = packed[pi] & 0xFF;
      sum += queryInt8[pi] * centroidInt8[p & 0x0F];
      sum += queryInt8[half + pi] * centroidInt8[(p >>> 4) & 0x0F];
    }
    return sum;
  }

  /** Dispatch to 1-bit or 2-bit bulk scorer. */
  private static float scorePolarBulk(QueryState query, TurboQuantVector doc, int dim, int bits) {
    int packedLen = TurboQuantVector.packedBinsBytes(dim, bits);
    byte[] packed = BIN_BUF.get();
    if (packed.length < packedLen) {
      packed = new byte[packedLen];
      BIN_BUF.set(packed);
    }
    MemorySegment.copy(
        doc.segment(),
        TurboQuantVector.PACKED_BINS_OFFSET,
        MemorySegment.ofArray(packed),
        0,
        packedLen);
    if (bits == 1) {
      return scorePolar1Bit(query, packed, dim);
    } else {
      return scorePolar2Bit(query.rotatedQuery, query.centroidLUT, packed, dim);
    }
  }

  /**
   * 1-bit scoring via 4-plane weighted popcount. Processes 32 dims per Integer.bitCount — same
   * throughput as BBQ.
   */
  // TODO(turboquant): SIMD optimization — use vectorized POPCNT (VectorOperators.ZOMO + AND + reduceLanes)
  private static float scorePolar1Bit(QueryState query, byte[] packed, int dim) {
    float[] lut = query.centroidLUT;
    float c0 = lut[0], diff = lut[1] - lut[0];
    byte[] qPlanes = query.queryInt8Planes;
    int stripeBytes = (dim + 7) >>> 3;
    int fullInts = stripeBytes >>> 2;

    int wp0 = 0, wp1 = 0, wp2 = 0, wp3 = 0, docPop = 0;
    int s0 = 0, s1 = stripeBytes, s2 = 2 * stripeBytes, s3 = 3 * stripeBytes;
    for (int ii = 0; ii < fullInts; ii++) {
      int off = ii << 2;
      int dInt =
          (packed[off] & 0xFF)
              | ((packed[off + 1] & 0xFF) << 8)
              | ((packed[off + 2] & 0xFF) << 16)
              | ((packed[off + 3] & 0xFF) << 24);
      docPop += Integer.bitCount(dInt);
      wp0 += Integer.bitCount(packInt(qPlanes, s0 + off) & dInt);
      wp1 += Integer.bitCount(packInt(qPlanes, s1 + off) & dInt);
      wp2 += Integer.bitCount(packInt(qPlanes, s2 + off) & dInt);
      wp3 += Integer.bitCount(packInt(qPlanes, s3 + off) & dInt);
    }
    int weightedPop = wp0 + (wp1 << 1) + (wp2 << 2) + (wp3 << 3);
    float sumQ1 = weightedPop * query.planeInvScale + query.planeBias * docPop;
    return c0 * query.totalRotatedQuery + diff * sumQ1;
  }

  private static int packInt(byte[] arr, int off) {
    return (arr[off] & 0xFF)
        | ((arr[off + 1] & 0xFF) << 8)
        | ((arr[off + 2] & 0xFF) << 16)
        | ((arr[off + 3] & 0xFF) << 24);
  }

  /** 2-bit scoring: expand to float centroids + scalar dot product. */
  // TODO(turboquant): SIMD optimization — vectorize 2-bit unpacking and LUT gather
  private static float scorePolar2Bit(float[] q, float[] lut, byte[] packed, int dim) {
    float sum = 0;
    int di = 0;
    for (int pi = 0; pi < packed.length && di < dim; pi++) {
      int b = packed[pi] & 0xFF;
      sum += q[di] * lut[b & 3];
      di++;
      if (di < dim) {
        sum += q[di] * lut[(b >>> 2) & 3];
        di++;
      }
      if (di < dim) {
        sum += q[di] * lut[(b >>> 4) & 3];
        di++;
      }
      if (di < dim) {
        sum += q[di] * lut[(b >>> 6) & 3];
        di++;
      }
    }
    return sum;
  }

  /** Float-precision polar scorer — generic for any bit width. */
  private static float scorePolarFloat(
      float[] rotatedQuery, float[] lut, TurboQuantVector doc, int dim) {
    int bits = doc.bits();
    float sum = 0;
    for (int i = 0; i < dim; i++) {
      sum += rotatedQuery[i] * lut[doc.getBin(i, bits) & 0xFF];
    }
    return sum;
  }

  /** Batch score multiple documents. */
  private static final ValueLayout.OfInt INT_LE =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN);

  /**
   * Signed int8 dot product: byte[] × MemorySegment. Matches Lucene's PanamaVectorUtilSupport
   * dotProductBody256 pattern: load 8 bytes via SPECIES_64, widen to 8 ints via single B2I into
   * SPECIES_256, multiply as int32, single accumulator. TODO(turboquant): propose this overload to VectorUtil
   * upstream (PanamaVectorUtilSupport already has the internal implementation).
   */
  static int dotProductInt8(byte[] a, MemorySegment b, long offset, int len) {
    var acc = IntVector.zero(IntVector.SPECIES_256);
    int i = 0;
    final int bound = ByteVector.SPECIES_64.loopBound(len);
    for (; i < bound; i += 8) {
      var va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
      var vb =
          ByteVector.fromMemorySegment(
              ByteVector.SPECIES_64, b, offset + i, java.nio.ByteOrder.LITTLE_ENDIAN);
      var ia = va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0).reinterpretAsInts();
      var ib = vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0).reinterpretAsInts();
      acc = acc.add(ia.mul(ib));
    }
    int sum = acc.reduceLanes(VectorOperators.ADD);
    for (; i < len; i++) {
      sum += a[i] * b.get(ValueLayout.JAVA_BYTE, offset + i);
    }
    return sum;
  }

  /** Expanded mode: zero-copy int8 dot product directly from MemorySegment. */
  static int scoreExpandedInt8Direct(byte[] queryInt8, MemorySegment data, long offset, int dim) {
    return dotProductInt8(queryInt8, data, offset, dim);
  }

  private static final ThreadLocal<byte[]> EXPAND_BUF =
      ThreadLocal.withInitial(() -> new byte[4096]);

  /**
   * 4-bit float LUT scorer — reads packed nibbles directly from MemorySegment, accumulates
   * rotatedQuery[i] * centroidLUT[bin] in float. No intermediate array, no int8 quantization loss.
   * The 16-entry float LUT fits in L1 cache for fast random access.
   *
   * <p>Optimization history (benchmarked on 5K×1024d, Graviton3 aarch64):
   *
   * <ol>
   *   <li>SIMD rearrange + int8 dot (4670ms): ByteVector.rearrange expands nibbles to int8
   *       centroid values in an intermediate byte[], then VectorUtil.dotProduct scores. Two-pass
   *       over data and int8 quantization of the query loses precision.
   *   <li>Fused SIMD castShape (5385ms): Attempted to fuse expand+accumulate in SIMD registers
   *       via castShape(byte→int). Slower — castShape generates expensive cross-lane shuffles on
   *       aarch64 (NEON has no single-instruction byte→int widening for 16 lanes).
   *   <li>Bin accumulation (4625ms): Accumulate query values by bin (binSum[bin] += query[i]),
   *       then 16 final multiplies. Slower — scatter-add to binSum[nibble] causes pipeline stalls
   *       due to data-dependent array index.
   *   <li><b>Direct float LUT + bulk read (4491ms, current):</b> Bulk MemorySegment.copy to local
   *       byte[], then scalar loop with float LUT lookup. Simplest code, highest precision (no
   *       int8 quantization), and fastest. The ~1.4× gap to SQ-4bit (3147ms) is inherent to
   *       LUT-based scoring vs SQ's arithmetic scoring (subtract+multiply, fully SIMD-pipelined).
   * </ol>
   */
  static float scorePolar4BitFloatDirect(
      float[] rotatedQuery, float[] centroidLUT, MemorySegment data, long offset,
      int packedLen, int dim) {
    byte[] packed = BIN_BUF.get();
    if (packed.length < packedLen) {
      packed = new byte[packedLen];
      BIN_BUF.set(packed);
    }
    MemorySegment.copy(data, offset, MemorySegment.ofArray(packed), 0, packedLen);
    float sum = 0;
    int di = 0;
    for (int pi = 0; pi < packedLen; pi++) {
      int p = packed[pi] & 0xFF;
      sum += rotatedQuery[di++] * centroidLUT[p & 0x0F];
      sum += rotatedQuery[di++] * centroidLUT[p >>> 4];
    }
    return sum;
  }

  /**
   * Hybrid int8 polar scorer for packed 4-bit bins. Phase 1: SIMD nibble expand via
   * ByteVector.rearrange (NEON TBL / x86 PSHUFB) to deinterleaved centroid int8 byte[]. Phase 2:
   * {@link VectorUtil#dotProduct(byte[], byte[])} for platform-tuned SIMD dot product.
   */
  static int scorePolarInt8Direct(
      byte[] queryInt8, byte[] centroidInt8, MemorySegment data, long offset, int packedLen) {
    int dim = queryInt8.length;
    byte[] expanded = EXPAND_BUF.get();
    if (expanded.length != dim) {
      expanded = new byte[dim];
      EXPAND_BUF.set(expanded);
    }
    // Phase 1: SIMD nibble expand via rearrange (NEON TBL / x86 PSHUFB)
    var centroidTable = ByteVector.fromArray(B128, centroidInt8, 0);
    final byte LO_MASK = 0x0F;
    int pi = 0;
    final int bound = B128.loopBound(packedLen);
    for (; pi < bound; pi += 16) {
      var packed =
          ByteVector.fromMemorySegment(B128, data, offset + pi, java.nio.ByteOrder.LITTLE_ENDIAN);
      centroidTable.rearrange(packed.and(LO_MASK).toShuffle()).intoArray(expanded, pi);
      centroidTable
          .rearrange(packed.lanewise(VectorOperators.LSHR, 4).and(LO_MASK).toShuffle())
          .intoArray(expanded, packedLen + pi);
    }
    for (; pi < packedLen; pi++) {
      int p = data.get(ValueLayout.JAVA_BYTE, offset + pi) & 0xFF;
      expanded[pi] = centroidInt8[p & 0x0F];
      expanded[packedLen + pi] = centroidInt8[p >>> 4];
    }
    // Phase 2: platform-tuned int8 dot product
    return VectorUtil.dotProduct(queryInt8, expanded);
  }

  /**
   * 8-bit packed scorer. Expands bins via centroid LUT to byte[], then delegates to {@link
   * VectorUtil#dotProduct(byte[], byte[])}.
   */
  static int scorePolar8BitInt8Direct(
      byte[] queryInt8, byte[] centroidInt8, MemorySegment data, long offset, int dim) {
    byte[] expanded = EXPAND_BUF.get();
    if (expanded.length != dim) {
      expanded = new byte[dim];
      EXPAND_BUF.set(expanded);
    }
    for (int i = 0; i < dim; i++) {
      expanded[i] = centroidInt8[data.get(ValueLayout.JAVA_BYTE, offset + i) & 0xFF];
    }
    return VectorUtil.dotProduct(queryInt8, expanded);
  }

  /**
   * 1-bit scoring directly from MemorySegment — zero-copy, unrolled 4-plane popcount. Uses int[]
   * query planes to avoid byte-to-int assembly in the hot loop.
   */
  static float scorePolar1BitDirect(
      QueryState query, MemorySegment data, long offset, int packedLen, int dim) {
    float[] lut = query.centroidLUT;
    float c0 = lut[0], diff = lut[1] - lut[0];
    int[] qp = query.queryPlanesInt;
    int intsPerStripe = (dim + 31) >>> 5;

    int wp0 = 0, wp1 = 0, wp2 = 0, wp3 = 0, docPop = 0;
    int s1 = intsPerStripe, s2 = 2 * intsPerStripe, s3 = 3 * intsPerStripe;
    for (int ii = 0; ii < intsPerStripe; ii++) {
      int dInt = data.get(INT_LE, offset + ((long) ii << 2));
      docPop += Integer.bitCount(dInt);
      wp0 += Integer.bitCount(qp[ii] & dInt);
      wp1 += Integer.bitCount(qp[s1 + ii] & dInt);
      wp2 += Integer.bitCount(qp[s2 + ii] & dInt);
      wp3 += Integer.bitCount(qp[s3 + ii] & dInt);
    }
    int weightedPop = wp0 + (wp1 << 1) + (wp2 << 2) + (wp3 << 3);
    float sumQ1 = weightedPop * query.planeInvScale + query.planeBias * docPop;
    return c0 * query.totalRotatedQuery + diff * sumQ1;
  }
}
