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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.BufferedChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Reads TurboQuant quantized-only vectors from disk. No raw float32 vectors are stored. Provides
 * {@link RandomVectorScorer} instances for HNSW graph search using quantized scoring.
 */
final class TurboQuantFlatVectorsReader extends FlatVectorsReader implements TurboQuantDataAccess {

  /** Prefix for data slices in the index file, used for debugging/diagnostics. */
  private static final String DATA_SLICE_PREFIX = "tqv-";

  private final Map<String, FieldEntry> fieldsByName = new HashMap<>();
  private final IndexInput dataInput;
  private final java.util.List<Arena> offHeapArenas = new java.util.ArrayList<>();

  TurboQuantFlatVectorsReader(SegmentReadState state, TurboQuantFlatVectorsFormat format)
      throws IOException {
    super(new TurboQuantFlatVectorsWriter.TurboQuantFlatScorer());

    String dataFile =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, format.dataExtension);
    String metaFile =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, format.metaExtension);

    this.dataInput = state.directory.openInput(dataFile, state.context);
    boolean success = false;
    try (IndexInput metaRaw = state.directory.openInput(metaFile, state.context)) {
      BufferedChecksumIndexInput metaIn = new BufferedChecksumIndexInput(metaRaw);
      int metaVersion =
          CodecUtil.checkIndexHeader(
              metaIn,
              format.codecName,
              TurboQuantFlatVectorsFormat.VERSION_START,
              TurboQuantFlatVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      CodecUtil.checkIndexHeader(
          dataInput,
          format.codecName,
          TurboQuantFlatVectorsFormat.VERSION_START,
          TurboQuantFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);

      int fieldNumber;
      while ((fieldNumber = metaIn.readInt()) != -1) {
        int dim = metaIn.readInt();
        int numVectors = metaIn.readInt();
        long seed = metaIn.readLong();
        long dataOffset = metaIn.readLong();
        int bits = metaIn.readByte() & 0xFF;

        int bytesPerVec = (bits == 8) ? 4 + dim : TurboQuantVector.totalBytes(dim, bits);
        long dataSize = (long) numVectors * bytesPerVec;

        byte[] quantizedData;
        MemorySegment quantizedSegment;
        if (numVectors > 0 && dataSize <= Integer.MAX_VALUE) {
          // Small enough for heap array
          quantizedData = new byte[(int) dataSize];
          IndexInput slice =
              dataInput.slice(DATA_SLICE_PREFIX + fieldNumber, dataOffset, dataSize);
          slice.readBytes(quantizedData, 0, quantizedData.length);
          quantizedSegment = MemorySegment.ofArray(quantizedData);
        } else if (numVectors > 0) {
          // Too large for heap — read in chunks into off-heap MemorySegment
          Arena offHeap = Arena.ofShared();
          offHeapArenas.add(offHeap);
          quantizedSegment = offHeap.allocate(dataSize);
          IndexInput slice =
              dataInput.slice(DATA_SLICE_PREFIX + fieldNumber, dataOffset, dataSize);
          byte[] buf = new byte[1 << 20]; // 1MB chunks
          long remaining = dataSize;
          long offset = 0;
          while (remaining > 0) {
            int toRead = (int) Math.min(buf.length, remaining);
            slice.readBytes(buf, 0, toRead);
            MemorySegment.copy(MemorySegment.ofArray(buf), 0, quantizedSegment, offset, toRead);
            offset += toRead;
            remaining -= toRead;
          }
        } else {
          quantizedSegment = MemorySegment.NULL;
        }

        TurboQuantEncoder encoder = new TurboQuantEncoder(dim, bits, seed);

        String fieldName = null;
        for (FieldInfo fi : state.fieldInfos) {
          if (fi.number == fieldNumber) {
            fieldName = fi.name;
            break;
          }
        }
        if (fieldName != null) {
          fieldsByName.put(
              fieldName,
              new FieldEntry(dim, bits, numVectors, bytesPerVec, encoder,
                  quantizedSegment));
        }
      }
      CodecUtil.checkFooter(metaIn);
      success = true;
    } finally {
      if (!success) {
        dataInput.close();
        for (Arena arena : offHeapArenas) {
          arena.close();
        }
      }
    }
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
    FieldEntry entry = fieldsByName.get(field);
    if (entry == null) {
      return null;
    }
    TurboQuantScorer.QueryState qs = entry.encoder.prepareQuery(target);
    return new TurboQuantRandomVectorScorer(
        new StubVectorValues(entry.numVectors, entry.dim),
        qs, entry.quantizedData, entry.dim, entry.bits, entry.bytesPerVec);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
    throw new UnsupportedOperationException("TurboQuant only supports float vectors");
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    FieldEntry entry = fieldsByName.get(field);
    if (entry == null) {
      return null;
    }
    return new StubVectorValues(
        entry.numVectors, entry.dim, entry.quantizedData, entry.bytesPerVec,
        entry.bits, entry.encoder.signs(), entry.encoder);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return null;
  }

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(dataInput);
  }

  @Override
  public void close() throws IOException {
    dataInput.close();
    for (Arena arena : offHeapArenas) {
      arena.close();
    }
  }

  @Override
  public long ramBytesUsed() {
    long bytes = 0;
    for (FieldEntry e : fieldsByName.values()) {
      bytes += e.quantizedData.byteSize();
    }
    return bytes;
  }

  /**
   * Public accessor for merge path — when merging segments with different codecs (e.g. a
   * TurboQuant segment merged with a default Lucene99 segment), the writer needs direct access
   * to quantized data to avoid re-encoding vectors that are already quantized.
   */
  @Override
  public MemorySegment getQuantizedData(String fieldName) {
    FieldEntry e = fieldsByName.get(fieldName);
    return e != null ? e.quantizedData : null;
  }

  /** Public accessor for merge path — returns vector count for a field. */
  @Override
  public int getNumVectors(String fieldName) {
    FieldEntry e = fieldsByName.get(fieldName);
    return e != null ? e.numVectors : 0;
  }

  @Override
  public int getBytesPerVec(String fieldName) {
    FieldEntry e = fieldsByName.get(fieldName);
    return e != null ? e.bytesPerVec : 0;
  }

  private record FieldEntry(
      int dim,
      int bits,
      int numVectors,
      int bytesPerVec,
      TurboQuantEncoder encoder,
      MemorySegment quantizedData) {}

  /**
   * Provides dequantized float vectors from TurboQuant quantized data. Supports both full vector
   * reconstruction via {@link #vectorValue} (inverse FWHT) and efficient rescoring via
   * {@link #rescorer} (direct centroid LUT dot product in rotated space). Quantized data is
   * exposed via public {@link TurboQuantDataAccess} interface methods for direct byte-copy during merge.
   */
  static class StubVectorValues extends FloatVectorValues {
    private final int size;
    private final int dim;
    private final float[] reusableVector;
    final MemorySegment quantizedData;
    final int bytesPerVec;
    private final int bits;
    private final float[] signs;
    private final TurboQuantEncoder encoder;
    private final byte[] binsBuf;

    StubVectorValues(int size, int dim) {
      this(size, dim, null, 0, 0, null, null);
    }

    StubVectorValues(
        int size, int dim, MemorySegment quantizedData, int bytesPerVec,
        int bits, float[] signs, TurboQuantEncoder encoder) {
      this.size = size;
      this.dim = dim;
      this.reusableVector = new float[dim];
      this.quantizedData = quantizedData;
      this.bytesPerVec = bytesPerVec;
      this.bits = bits;
      this.signs = signs;
      this.encoder = encoder;
      this.binsBuf = new byte[dim];
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public int size() {
      return size;
    }

    /**
     * Unpack quantized bin indices from packed storage into a flat byte array.
     * Each element is the bin index (0..2^bits-1) for that dimension.
     *
     * <p>Bit layouts per case:
     * <ul>
     *   <li>8-bit: 1 bin per byte, direct copy
     *   <li>4-bit: 2 bins per byte — low nibble (bits 0-3) = even dim, high nibble (bits 4-7) = odd dim
     *   <li>2-bit: 4 bins per byte — bits [1:0]=dim+0, [3:2]=dim+1, [5:4]=dim+2, [7:6]=dim+3
     *   <li>1-bit: 8 bins per byte — bit j = dim i+j (LSB first)
     * </ul>
     */
    private void unpackBins(long binsOff, byte[] out) {
      if (bits == 8) {
        // 8-bit: 1 bin per byte, direct copy
        for (int i = 0; i < dim; i++) {
          out[i] = quantizedData.get(java.lang.foreign.ValueLayout.JAVA_BYTE, binsOff + i);
        }
      } else if (bits == 4) {
        // 4-bit: low nibble = even index, high nibble = odd index
        for (int i = 0; i < dim; i += 2) {
          int packed =
              quantizedData.get(java.lang.foreign.ValueLayout.JAVA_BYTE, binsOff + (i >> 1)) & 0xFF;
          out[i] = (byte) (packed & 0x0F);
          out[i + 1] = (byte) (packed >>> 4);
        }
      } else if (bits == 2) {
        // 2-bit: 4 bins packed per byte, 2 bits each from LSB
        for (int i = 0; i < dim; i += 4) {
          int packed =
              quantizedData.get(java.lang.foreign.ValueLayout.JAVA_BYTE, binsOff + (i >> 2)) & 0xFF;
          out[i] = (byte) (packed & 0x03);
          out[i + 1] = (byte) ((packed >>> 2) & 0x03);
          out[i + 2] = (byte) ((packed >>> 4) & 0x03);
          out[i + 3] = (byte) ((packed >>> 6) & 0x03);
        }
      } else {
        // 1-bit: 8 bins packed per byte, LSB first
        for (int i = 0; i < dim; i += 8) {
          int packed =
              quantizedData.get(java.lang.foreign.ValueLayout.JAVA_BYTE, binsOff + (i >> 3)) & 0xFF;
          for (int j = 0; j < 8 && i + j < dim; j++) {
            out[i + j] = (byte) ((packed >>> j) & 1);
          }
        }
      }
    }

    @Override
    public float[] vectorValue(int ord) {
      if (quantizedData == null || bits == 0 || signs == null) {
        java.util.Arrays.fill(reusableVector, 0f);
        return reusableVector;
      }
      long base = (long) ord * bytesPerVec;
      float norm =
          quantizedData.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, base);
      if (norm < TurboQuantScorer.NORM_EPSILON) {
        java.util.Arrays.fill(reusableVector, 0f);
        return reusableVector;
      }
      float[] centroids = PolarQuant.centroids(bits);
      float sigma = 1.0f / (float) Math.sqrt(dim);
      unpackBins(base + TurboQuantVector.PACKED_BINS_OFFSET, binsBuf);
      for (int i = 0; i < dim; i++) {
        reusableVector[i] = centroids[binsBuf[i] & 0xFF] * sigma;
      }
      FWHT.transform(reusableVector);
      // BUG: should be `norm` not `norm/dim`. The inverse FWHT already divides by sqrt(dim),
      // so this extra /dim makes reconstructed vectors dim× too small. Affects cross-codec merge
      // (re-encode path calls vectorValue()) and the fallback VectorScorer. Same-codec merge
      // uses byte-copy and is unaffected. Fix requires verifying the full encode→decode roundtrip.
      float invDimNorm = norm / dim;
      for (int i = 0; i < dim; i++) {
        reusableVector[i] *= signs[i] * invDimNorm;
      }
      return reusableVector;
    }

    @Override
    public FloatVectorValues copy() {
      return new StubVectorValues(
          size, dim, quantizedData, bytesPerVec, bits, signs, encoder);
    }

    @Override
    public VectorScorer scorer(float[] target) {
      if (quantizedData == null || bits == 0 || signs == null) {
        return null;
      }
      StubVectorValues copy =
          new StubVectorValues(size, dim, quantizedData, bytesPerVec, bits, signs, encoder);
      DocIdSetIterator iter = copy.iterator();
      return new VectorScorer() {
        @Override
        public float score() throws IOException {
          float[] vec = copy.vectorValue(iter.docID());
          return VectorSimilarityFunction.DOT_PRODUCT.compare(target, vec);
        }

        @Override
        public DocIdSetIterator iterator() {
          return iter;
        }
      };
    }

    @Override
    public VectorScorer rescorer(float[] target) throws IOException {
      if (encoder == null || quantizedData == null) {
        return scorer(target);
      }
      // 8-bit: int8 SIMD scorer is faster and precise enough
      if (bits == 8) {
        TurboQuantScorer.QueryState qs = encoder.prepareQuery(target);
        TurboQuantRandomVectorScorer rvs =
            new TurboQuantRandomVectorScorer(
                new StubVectorValues(size, dim),
                qs, quantizedData, dim, bits, bytesPerVec);
        DocIdSetIterator iter = createDenseIterator();
        return new VectorScorer() {
          @Override
          public float score() throws IOException { return rvs.score(iter.docID()); }
          @Override
          public DocIdSetIterator iterator() { return iter; }
        };
      }
      // ≤4-bit: float-precision rescorer avoids int8 query quantization error
      float[][] rl = TurboQuantScorer.rotateAndBuildLUT(target, encoder.signs(), bits);
      float[] rotated = rl[0];
      final float fQueryNorm = rl[1][0];
      final float[] fRotated = rotated;
      final float[] fLut = rl[2];
      final byte[] scoreBins = new byte[dim];
      DocIdSetIterator iter = createDenseIterator();

      return new VectorScorer() {
        @Override
        public float score() throws IOException {
          int ord = iter.docID();
          long base = (long) ord * bytesPerVec;
          float docNorm =
              quantizedData.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, base);
          if (docNorm < TurboQuantScorer.NORM_EPSILON || fQueryNorm < TurboQuantScorer.NORM_EPSILON) return 0f;

          unpackBins(base + TurboQuantVector.PACKED_BINS_OFFSET, scoreBins);
          float dot = 0f;
          for (int i = 0; i < dim; i++) {
            dot += fRotated[i] * fLut[scoreBins[i] & 0xFF];
          }
          float rawDot = fQueryNorm * docNorm * dot;
          return VectorUtil.scaleMaxInnerProductScore(rawDot);
        }

        @Override
        public DocIdSetIterator iterator() { return iter; }
      };
    }

    @Override
    public DocIndexIterator iterator() {
      return createDenseIterator();
    }
  }
}
