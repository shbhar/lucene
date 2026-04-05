/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.
 */
package org.apache.lucene.codecs.turboquant;

import java.lang.foreign.Arena;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Tests for Lloyd-Max polar quantization. */
public class TestPolarQuant extends LuceneTestCase {

  public void testCentroidsAreSymmetric() {
    for (int bits : new int[]{1, 2, 4, 8}) {
      float[] c = PolarQuant.centroids(bits);
      assertEquals(1 << bits, c.length);
      // Centroids should be symmetric around 0 for the standard normal distribution
      for (int i = 0; i < c.length / 2; i++) {
        assertEquals(-c[i], c[c.length - 1 - i], 1e-4f);
      }
    }
  }

  public void testCentroidsAreOrdered() {
    for (int bits : new int[]{1, 2, 4, 8}) {
      float[] c = PolarQuant.centroids(bits);
      for (int i = 1; i < c.length; i++) {
        assertTrue("centroids must be strictly increasing", c[i] > c[i - 1]);
      }
    }
  }

  public void testEncodeDecodeRoundTrip() {
    int dim = 256;
    TurboQuantEncoder enc = new TurboQuantEncoder(dim, 4, 42L);
    float[] vec = randomUnitVector(dim);
    try (Arena arena = Arena.ofConfined()) {
      TurboQuantVector tqv = enc.encode(arena, vec);
      // Norm should be close to 1.0 for unit vector
      float norm = tqv.getNorm();
      assertEquals(1.0f, norm, 0.01f);
      // Bins should be in valid range [0, 15] for 4-bit
      for (int i = 0; i < dim; i++) {
        byte bin = tqv.getBin(i, 4);
        assertTrue(bin >= 0 && bin < 16);
      }
    }
  }

  public void testEncodePreservesNorm() {
    int dim = 512;
    for (int bits : new int[]{1, 2, 4, 8}) {
      TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, 42L);
      float[] vec = new float[dim];
      float n = 0;
      for (int i = 0; i < dim; i++) { vec[i] = (float) random().nextGaussian(); n += vec[i] * vec[i]; }
      n = (float) Math.sqrt(n);
      try (Arena arena = Arena.ofConfined()) {
        TurboQuantVector tqv = enc.encode(arena, vec);
        assertEquals(n, tqv.getNorm(), 0.01f);
      }
    }
  }

  public void testZeroVectorEncodesCleanly() {
    int dim = 128;
    TurboQuantEncoder enc = new TurboQuantEncoder(dim, 4, 42L);
    float[] zero = new float[dim];
    try (Arena arena = Arena.ofConfined()) {
      TurboQuantVector tqv = enc.encode(arena, zero);
      assertEquals(0f, tqv.getNorm(), 1e-6f);
    }
  }

  public void testDifferentSeedsProduceDifferentEncodings() {
    int dim = 256;
    float[] vec = randomUnitVector(dim);
    try (Arena arena = Arena.ofConfined()) {
      TurboQuantVector a = new TurboQuantEncoder(dim, 4, 1L).encode(arena, vec);
      TurboQuantVector b = new TurboQuantEncoder(dim, 4, 2L).encode(arena, vec);
      // Different seeds → different FWHT signs → different bins
      boolean anyDiff = false;
      for (int i = 0; i < dim; i++) {
        if (a.getBin(i, 4) != b.getBin(i, 4)) { anyDiff = true; break; }
      }
      assertTrue("different seeds should produce different encodings", anyDiff);
    }
  }

  public void testInvalidBitsThrows() {
    expectThrows(IllegalArgumentException.class, () -> new TurboQuantEncoder(128, 3, 42L));
  }

  public void testNonPowerOf2DimThrows() {
    expectThrows(IllegalArgumentException.class, () -> new TurboQuantEncoder(100, 4, 42L));
  }

  public void testQuantizeDequantizeRoundTrip() {
    for (int bits : new int[]{1, 2, 4, 8}) {
      float[] centroids = PolarQuant.centroids(bits);
      for (int bin = 0; bin < centroids.length; bin++) {
        float centroid = centroids[bin];
        int quantized = PolarQuant.quantize(centroid, bits);
        float dequantized = PolarQuant.dequantize(quantized, bits);
        assertEquals(bits + "-bit bin " + bin + " round-trip", centroid, dequantized, 1e-3f);
      }
    }
  }

  private float[] randomUnitVector(int dim) {
    float[] v = new float[dim];
    float n = 0;
    for (int i = 0; i < dim; i++) { v[i] = (float) random().nextGaussian(); n += v[i] * v[i]; }
    n = (float) Math.sqrt(n);
    for (int i = 0; i < dim; i++) v[i] /= n;
    return v;
  }
}
