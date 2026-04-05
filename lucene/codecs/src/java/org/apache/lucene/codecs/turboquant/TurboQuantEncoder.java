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

import java.lang.foreign.Arena;
import java.util.SplittableRandom;

/**
 * Encoder: FWHT rotation + Lloyd-Max polar quantization.
 *
 * <p>Encodes float32 vectors into compact quantized representations. The encoding is stateless —
 * each vector is quantized independently using global constants derived from the random seed.
 *
 * @lucene.experimental
 */
public final class TurboQuantEncoder {

  private final int dim;
  private final int bits;
  private final float[] signs;

  /**
   * @param dim vector dimension (must be power of 2)
   * @param bits quantization bit depth (1, 2, 4, or 8)
   * @param seed random seed for reproducible FWHT sign flips
   */
  public TurboQuantEncoder(int dim, int bits, long seed) {
    if (Integer.bitCount(dim) != 1) {
      throw new IllegalArgumentException("dim must be power of 2, got " + dim);
    }
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) {
      throw new IllegalArgumentException("bits must be 1, 2, 4, or 8, got " + bits);
    }
    this.dim = dim;
    this.bits = bits;
    SplittableRandom rng = new SplittableRandom(seed);
    this.signs = new float[dim];
    for (int i = 0; i < dim; i++) {
      signs[i] = rng.nextBoolean() ? 1.0f : -1.0f;
    }
  }

  /**
   * Encode a vector. Input array is NOT modified.
   *
   * @param arena memory arena for off-heap allocation of the quantized vector
   * @param vector input float vector of length dim
   * @return quantized TurboQuantVector with norm and packed bins
   */
  public TurboQuantVector encode(Arena arena, float[] vector) {
    if (vector.length != dim) {
      throw new IllegalArgumentException("Expected dim=" + dim + ", got " + vector.length);
    }
    TurboQuantVector target = TurboQuantVector.allocate(arena, dim, bits);
    float[] work = vector.clone();
    PolarQuant.encode(work, signs, target, bits);
    return target;
  }

  /**
   * Prepare a query for repeated scoring against quantized vectors.
   *
   * @param query input query vector (not modified)
   * @return precomputed query state for use with {@link TurboQuantScorer#score}
   */
  public TurboQuantScorer.QueryState prepareQuery(float[] query) {
    return TurboQuantScorer.prepareQuery(query, signs, bits);
  }

  /**
   * Rotate a query vector into the quantized coordinate space: normalize → sign-flip → FWHT.
   * Delegates to {@link TurboQuantScorer#rotateAndBuildLUT} to avoid divergence.
   *
   * @param query input query vector (not modified)
   * @return [rotated, queryNorm] where rotated is the transformed unit vector
   */
  public float[][] rotateQuery(float[] query) {
    float[][] rl = TurboQuantScorer.rotateAndBuildLUT(query, signs, bits);
    return new float[][] {rl[0], rl[1]};
  }

  /** @return vector dimension */
  public int dim() { return dim; }
  /** @return quantization bit depth */
  public int bits() { return bits; }
  /** @return random sign flip array used for FWHT rotation */
  public float[] signs() { return signs; }
}
