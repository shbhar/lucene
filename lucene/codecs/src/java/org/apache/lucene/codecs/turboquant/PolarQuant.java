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

/**
 * PolarQuant: Lloyd-Max scalar quantization of FWHT-rotated coordinates.
 *
 * <p>Implements the core quantizer from "TurboQuant: Online Vector Quantization with Near-optimal
 * Distortion Rate" (Zandieh et al., ICLR 2026, arxiv:2504.19874). This is the primary compression
 * stage of the TurboQuant system.
 *
 * <p>After randomized Hadamard rotation, each coordinate of a unit vector follows approximately
 * N(0, 1/sqrt(d)). We scale by sqrt(d) to get N(0,1) and apply Lloyd-Max optimal quantization at
 * the configured bit depth.
 *
 * <p>Supports 1, 2, 4, and 8 bits (2, 4, 16, 256 levels).
 *
 * @lucene.experimental
 */
public final class PolarQuant {

  private PolarQuant() {}

  /** Supported bit depths. */
  public static final int[] SUPPORTED_BITS = {1, 2, 4, 8};

  /**
   * Maximum Lloyd-Max iterations. 200 is sufficient for convergence to machine precision
   * for all supported bit depths (1-8 bit, 2-256 levels).
   */
  private static final int LLOYD_MAX_ITERATIONS = 200;

  /**
   * Integration bound in standard deviations. ±6σ covers 99.9999998% of the Gaussian
   * distribution, ensuring negligible truncation error in conditional mean computation.
   */
  private static final float INTEGRATION_BOUND = 6.0f;

  private static final float[][] ALL_BOUNDARIES = new float[9][];
  private static final float[][] ALL_CENTROIDS = new float[9][];

  static {
    for (int bits : SUPPORTED_BITS) {
      int levels = 1 << bits;
      float[] c = computeCentroids(levels);
      float[] b = computeBoundaries(c);
      ALL_CENTROIDS[bits] = c;
      ALL_BOUNDARIES[bits] = b;
    }
  }

  /**
   * Get centroids for the given bit depth.
   *
   * @param bits quantization bit depth (1, 2, 4, or 8)
   * @return array of centroid values, length 2^bits, in ascending order
   */
  public static float[] centroids(int bits) {
    validateBits(bits);
    return ALL_CENTROIDS[bits];
  }

  /**
   * Get boundaries for the given bit depth.
   *
   * @param bits quantization bit depth (1, 2, 4, or 8)
   * @return array of decision boundaries, length 2^bits - 1
   */
  public static float[] boundaries(int bits) {
    validateBits(bits);
    return ALL_BOUNDARIES[bits];
  }

  /**
   * Number of quantization levels for given bits.
   *
   * @param bits quantization bit depth
   * @return 2^bits
   */
  public static int levels(int bits) {
    return 1 << bits;
  }

  /**
   * Quantize a normalized value (x * sqrt(dim)) to bin index [0, levels).
   *
   * @param normalizedValue the scaled coordinate value
   * @param bits quantization bit depth (1, 2, 4, or 8)
   * @return bin index in [0, 2^bits)
   */
  public static int quantize(float normalizedValue, int bits) {
    float[] b = ALL_BOUNDARIES[bits];
    for (int i = 0; i < b.length; i++) {
      if (normalizedValue < b[i]) return i;
    }
    return levels(bits) - 1;
  }

  /**
   * Dequantize bin index to reconstruction value.
   *
   * @param bin quantization bin index
   * @param bits quantization bit depth (1, 2, 4, or 8)
   * @return the centroid value for the given bin
   */
  public static float dequantize(int bin, int bits) {
    return ALL_CENTROIDS[bits][bin];
  }

  /**
   * Encode: normalize, apply sign flips + FWHT, quantize each coordinate. Input vec is modified
   * in-place. Uses default 4-bit depth.
   *
   * @param vec input vector (modified in-place by rotation)
   * @param signs random sign flip array from encoder seed
   * @param target output TurboQuantVector to write bins and norm into
   */
  public static void encode(float[] vec, float[] signs, TurboQuantVector target) {
    encode(vec, signs, target, 4);
  }

  /**
   * Encode at specified bit depth.
   *
   * @param vec input vector (modified in-place by rotation)
   * @param signs random sign flip array from encoder seed
   * @param target output TurboQuantVector to write bins and norm into
   * @param bits quantization bit depth (1, 2, 4, or 8)
   */
  public static void encode(float[] vec, float[] signs, TurboQuantVector target, int bits) {
    final int dim = vec.length;
    float norm = 0;
    for (float v : vec) norm += v * v;
    norm = (float) Math.sqrt(norm);
    target.setNorm(norm);

    int numLevels = levels(bits);
    if (norm < 1e-30f) {
      for (int i = 0; i < dim; i++) target.setBin(i, (byte) (numLevels / 2), bits);
      return;
    }

    float invNorm = 1.0f / norm;
    for (int i = 0; i < dim; i++) vec[i] *= invNorm;
    FWHT.applySignFlips(vec, signs);
    FWHT.transform(vec);

    float scale = (float) Math.sqrt(dim);
    for (int i = 0; i < dim; i++) {
      target.setBin(i, (byte) quantize(vec[i] * scale, bits), bits);
    }
  }

  private static float[] computeCentroids(int levels) {
    float[] c = new float[levels];
    // Initial centroids: uniformly spaced in [-3, +3] (covers ±3σ of N(0,1)).
    // This symmetric seeding ensures Lloyd-Max converges to the globally optimal
    // quantizer for the Gaussian source — uniform spacing within ±3σ places centroids
    // where >99.7% of probability mass lies, avoiding empty-cell degenerate solutions.
    for (int i = 0; i < levels; i++) {
      c[i] = -3.0f + 6.0f * (i + 0.5f) / levels;
    }
    float[] b = new float[levels - 1];
    for (int iter = 0; iter < LLOYD_MAX_ITERATIONS; iter++) {
      for (int i = 0; i < levels - 1; i++) {
        b[i] = 0.5f * (c[i] + c[i + 1]);
      }
      for (int i = 0; i < levels; i++) {
        float lo = (i == 0) ? -INTEGRATION_BOUND : b[i - 1];
        float hi = (i == levels - 1) ? INTEGRATION_BOUND : b[i];
        c[i] = conditionalMean(lo, hi);
      }
    }
    return c;
  }

  private static float[] computeBoundaries(float[] c) {
    float[] b = new float[c.length - 1];
    for (int i = 0; i < b.length; i++) {
      b[i] = 0.5f * (c[i] + c[i + 1]);
    }
    return b;
  }

  private static float conditionalMean(float lo, float hi) {
    double pLo = Math.exp(-0.5 * lo * lo) / Math.sqrt(2 * Math.PI);
    double pHi = Math.exp(-0.5 * hi * hi) / Math.sqrt(2 * Math.PI);
    double cLo = 0.5 * (1.0 + erf(lo / Math.sqrt(2.0)));
    double cHi = 0.5 * (1.0 + erf(hi / Math.sqrt(2.0)));
    double denom = cHi - cLo;
    if (denom < 1e-15) return (float) (0.5 * (lo + hi));
    return (float) ((pLo - pHi) / denom);
  }

  /**
   * Error function approximation using Abramowitz & Stegun formula 7.1.26. Max error ~1.5×10⁻⁷.
   * Coefficients: a₁=0.254829592, a₂=-0.284496736, a₃=1.421413741, a₄=-1.453152027, a₅=1.061405429,
   * p=0.3275911.
   */
  private static double erf(double x) {
    double sign = (x >= 0) ? 1.0 : -1.0;
    double absX = Math.abs(x);
    double t = 1.0 / (1.0 + 0.3275911 * absX);
    double y =
        1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                    + 0.254829592)
                * t
                * Math.exp(-absX * absX);
    return sign * y;
  }

  private static void validateBits(int bits) {
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) {
      throw new IllegalArgumentException(
          "Unsupported bit depth: " + bits + ". Must be 1, 2, 4, or 8.");
    }
  }
}
