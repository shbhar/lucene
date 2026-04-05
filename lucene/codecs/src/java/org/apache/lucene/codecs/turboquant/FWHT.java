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
 * In-place Fast Walsh-Hadamard Transform.
 *
 * <p>The FWHT is a random rotation (when combined with random sign flips) that only uses additions
 * and subtractions. Input length must be a power of 2. After transform, values are scaled by
 * 1/sqrt(n) for normalization.
 *
 * @lucene.experimental
 */
public final class FWHT {

  private FWHT() {}

  /** In-place normalized FWHT. Array length must be power-of-2. */
  public static void transform(float[] data) {
    final int n = data.length;
    assert (n & (n - 1)) == 0 : "FWHT requires power-of-2 length, got " + n;
    for (int h = 1; h < n; h <<= 1) {
      for (int i = 0; i < n; i += h << 1) {
        for (int j = 0; j < h; j++) {
          float a = data[i + j];
          float b = data[i + h + j];
          data[i + j] = a + b;
          data[i + h + j] = a - b;
        }
      }
    }
    final float scale = (float) (1.0 / Math.sqrt(n));
    for (int i = 0; i < n; i++) {
      data[i] *= scale;
    }
  }

  /**
   * Apply random sign flips (diagonal of ±1) before FWHT. This makes the rotation pseudo-random
   * (randomized Hadamard). signs[i] should be +1.0f or -1.0f.
   */
  public static void applySignFlips(float[] data, float[] signs) {
    for (int i = 0; i < data.length; i++) {
      data[i] *= signs[i];
    }
  }
}
