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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Scores quantized TurboQuant vectors against a precomputed query state. Reads from contiguous
 * memory at offset = ord * bytesPerVec. Zero-allocation hot path.
 *
 * <p>Scoring paths by bit width:
 *
 * <ul>
 *   <li>8-bit: direct int8 dot product via {@link TurboQuantScorer#scoreExpandedInt8Direct}
 *   <li>4-bit: int8 SIMD gather via {@link TurboQuantScorer#scorePolarInt8Direct}
 *   <li>1-bit: popcount via {@link TurboQuantScorer#scorePolar1BitDirect}
 *   <li>2-bit: float LUT via {@link TurboQuantScorer#score}
 * </ul>
 *
 * <p>All scores are clamped ≥ 0 to satisfy Lucene's scoring contract.
 */
final class TurboQuantRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {

  private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED;

  private final TurboQuantScorer.QueryState queryState;
  private final MemorySegment quantizedData;
  private final int bytesPerVec;
  private final int packedBinsLen;
  private final int dim;
  private final int bits;

  TurboQuantRandomVectorScorer(
      KnnVectorValues values,
      TurboQuantScorer.QueryState queryState,
      MemorySegment quantizedData,
      int dim,
      int bits,
      int bytesPerVec) {
    super(values);
    this.queryState = queryState;
    this.quantizedData = quantizedData;
    this.bytesPerVec = bytesPerVec;
    this.bits = bits;
    this.packedBinsLen = TurboQuantVector.packedBinsBytes(dim, bits);
    this.dim = dim;
  }

  @Override
  public float score(int node) throws IOException {
    long base = (long) node * bytesPerVec;
    float docNorm = quantizedData.get(FLOAT_LE, base);
    if (docNorm < TurboQuantScorer.NORM_EPSILON || queryState.queryNorm < TurboQuantScorer.NORM_EPSILON) {
      return 0f;
    }

    float rawDot;
    if (bits == 8) {
      int intDot =
          TurboQuantScorer.scoreExpandedInt8Direct(
              queryState.queryInt8, quantizedData, base + 4, dim);
      rawDot = queryState.queryNorm * docNorm * intDot * queryState.int8InvScale;
    } else if (bits == 4) {
      float polarCos =
          TurboQuantScorer.scorePolar4BitFloatDirect(
              queryState.rotatedQuery,
              queryState.centroidLUT,
              quantizedData,
              base + TurboQuantVector.PACKED_BINS_OFFSET,
              packedBinsLen,
              dim);
      rawDot = queryState.queryNorm * docNorm * polarCos;
    } else if (bits == 1) {
      long binsOff = base + TurboQuantVector.PACKED_BINS_OFFSET;
      float polarCos =
          TurboQuantScorer.scorePolar1BitDirect(
              queryState, quantizedData, binsOff, packedBinsLen, dim);
      rawDot = queryState.queryNorm * docNorm * polarCos;
    } else if (bits == 2) {
      MemorySegment vecSeg = quantizedData.asSlice(base, bytesPerVec);
      TurboQuantVector tqv = TurboQuantVector.wrap(vecSeg, dim, bits);
      rawDot = TurboQuantScorer.score(queryState, tqv);
    } else {
      MemorySegment vecSeg = quantizedData.asSlice(base, bytesPerVec);
      TurboQuantVector tqv = TurboQuantVector.wrap(vecSeg, dim, bits);
      rawDot = TurboQuantScorer.scoreFloat(queryState, tqv);
    }
    return VectorUtil.scaleMaxInnerProductScore(rawDot);
  }
}
