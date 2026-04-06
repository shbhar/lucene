/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.
 */
package org.apache.lucene.codecs.turboquant;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.apache.lucene.tests.util.LuceneTestCase;

/**
 * Focused tests for SIMD scoring paths and bit-twiddling, comparing against naive fp32 reference
 * implementations.
 */
public class TestTurboQuantScorer extends LuceneTestCase {

  private static final int[] DIMS = {128, 256, 1024, 4096};
  private static final int[] BIT_WIDTHS = {1, 2, 4, 8};

  /** score() (int8 SIMD path) must agree with scoreFloat() (float reference) within tolerance. */
  public void testScoreMatchesScoreFloat() throws Exception {
    for (int dim : DIMS) {
      for (int bits : BIT_WIDTHS) {
        long seed = random().nextLong();
        TurboQuantEncoder encoder = new TurboQuantEncoder(dim, bits, seed);
        float[] doc = randomUnitVector(dim);
        float[] query = randomUnitVector(dim);

        try (Arena arena = Arena.ofConfined()) {
          TurboQuantVector tqDoc = encoder.encode(arena, doc);
          TurboQuantScorer.QueryState qs =
              TurboQuantScorer.prepareQuery(query, encoder.signs(), bits);

          float simdScore = TurboQuantScorer.score(qs, tqDoc);
          float floatScore = TurboQuantScorer.scoreFloat(qs, tqDoc);

          // int8 quantization of query introduces error; wider tolerance at low bits/dims
          float tol = (bits <= 2 && dim <= 256) ? 0.3f : bits <= 2 ? 0.15f : 0.05f;
          assertEquals(
              dim + "d " + bits + "-bit: score vs scoreFloat",
              floatScore, simdScore, Math.abs(floatScore) * tol + 0.05f);
        }
      }
    }
  }

  /** Centroid LUT score must approximate true cosine similarity. */
  public void testScoreApproximatesCosine() throws Exception {
    for (int dim : DIMS) {
      for (int bits : BIT_WIDTHS) {
        long seed = random().nextLong();
        TurboQuantEncoder encoder = new TurboQuantEncoder(dim, bits, seed);
        float[] doc = randomUnitVector(dim);
        float[] query = randomUnitVector(dim);
        float trueCosine = dot(query, doc);

        try (Arena arena = Arena.ofConfined()) {
          TurboQuantVector tqDoc = encoder.encode(arena, doc);
          TurboQuantScorer.QueryState qs =
              TurboQuantScorer.prepareQuery(query, encoder.signs(), bits);
          float tqScore = TurboQuantScorer.scoreFloat(qs, tqDoc);

          float maxError = switch (bits) {
            case 1 -> 0.35f;
            case 2 -> 0.20f;
            case 4 -> 0.10f;
            case 8 -> 0.03f;
            default -> throw new AssertionError();
          };
          assertEquals(
              dim + "d " + bits + "-bit: score vs cosine",
              trueCosine, tqScore, maxError);
        }
      }
    }
  }

  /** Self-score (doc scored against itself) should be maximal and close to 1.0. */
  public void testSelfScoreIsMaximal() throws Exception {
    for (int dim : new int[]{256, 1024}) {
      for (int bits : BIT_WIDTHS) {
        long seed = random().nextLong();
        TurboQuantEncoder encoder = new TurboQuantEncoder(dim, bits, seed);
        float[] vec = randomUnitVector(dim);

        try (Arena arena = Arena.ofConfined()) {
          TurboQuantVector tqVec = encoder.encode(arena, vec);
          TurboQuantScorer.QueryState qs =
              TurboQuantScorer.prepareQuery(vec, encoder.signs(), bits);
          float selfScore = TurboQuantScorer.scoreFloat(qs, tqVec);

          // Self-score should be close to 1.0 (exact for 8-bit)
          float minSelf = bits >= 4 ? 0.9f : 0.5f;
          assertTrue(dim + "d " + bits + "-bit: selfScore=" + selfScore + " < " + minSelf,
              selfScore >= minSelf);
        }
      }
    }
  }

  /** dotProductInt8 must match scalar computation. */
  public void testDotProductInt8MatchesScalar() throws Exception {
    for (int len : new int[]{32, 128, 1024, 4096}) {
      byte[] a = new byte[len];
      byte[] b = new byte[len];
      random().nextBytes(a);
      random().nextBytes(b);

      // Scalar reference
      int expected = 0;
      for (int i = 0; i < len; i++) expected += a[i] * b[i];

      // SIMD path
      MemorySegment bSeg = MemorySegment.ofArray(b);
      int actual = TurboQuantScorer.dotProductInt8(a, bSeg, 0, len);

      assertEquals("dotProductInt8 at len=" + len, expected, actual);
    }
  }

  /** Bin packing roundtrip: pack → unpack must recover original bins. */
  public void testBinPackingRoundtrip() throws Exception {
    for (int dim : new int[]{128, 1024}) {
      for (int bits : BIT_WIDTHS) {
        int numLevels = 1 << bits;
        int[] bins = new int[dim];
        for (int i = 0; i < dim; i++) bins[i] = random().nextInt(numLevels);

        try (Arena arena = Arena.ofConfined()) {
          TurboQuantVector tqv = TurboQuantVector.allocate(arena, dim, bits);
          // Pack bins
          MemorySegment seg = tqv.segment();
          int packedLen = TurboQuantVector.packedBinsBytes(dim, bits);
          byte[] packed = new byte[packedLen];
          packBins(bins, packed, bits);
          for (int i = 0; i < packedLen; i++)
            seg.set(java.lang.foreign.ValueLayout.JAVA_BYTE, TurboQuantVector.PACKED_BINS_OFFSET + i, packed[i]);

          // Unpack and verify
          for (int i = 0; i < dim; i++) {
            int recovered = unpackBin(packed, i, bits);
            assertEquals(dim + "d " + bits + "-bit bin[" + i + "]", bins[i], recovered);
          }
        }
      }
    }
  }

  /** Higher bit width must produce score closer to true cosine on average. */
  public void testHigherBitsImproveAccuracy() throws Exception {
    for (int dim : new int[]{256, 1024, 4096}) {
      long seed = random().nextLong();
      int nTrials = 50;

      float prevAvgError = Float.MAX_VALUE;
      for (int bits : BIT_WIDTHS) {
        float totalError = 0;
        for (int t = 0; t < nTrials; t++) {
          float[] doc = randomUnitVector(dim);
          float[] query = randomUnitVector(dim);
          float trueCosine = dot(query, doc);
          TurboQuantEncoder encoder = new TurboQuantEncoder(dim, bits, seed);
          try (Arena arena = Arena.ofConfined()) {
            TurboQuantVector tqDoc = encoder.encode(arena, doc);
            TurboQuantScorer.QueryState qs =
                TurboQuantScorer.prepareQuery(query, encoder.signs(), bits);
            float score = TurboQuantScorer.scoreFloat(qs, tqDoc);
            totalError += Math.abs(score - trueCosine);
          }
        }
        float avgError = totalError / nTrials;
        assertTrue(dim + "d: " + bits + "-bit avgError=" + avgError + " >= prev=" + prevAvgError,
            avgError <= prevAvgError + 0.005f);
        prevAvgError = avgError;
      }
    }
  }

  // --- helpers ---

  private float[] randomUnitVector(int dim) {
    float[] v = new float[dim];
    float n = 0;
    for (int i = 0; i < dim; i++) { v[i] = (float) random().nextGaussian(); n += v[i] * v[i]; }
    n = (float) Math.sqrt(n);
    for (int i = 0; i < dim; i++) v[i] /= n;
    return v;
  }

  private static float dot(float[] a, float[] b) {
    float d = 0;
    for (int i = 0; i < a.length; i++) d += a[i] * b[i];
    return d;
  }

  private static void packBins(int[] bins, byte[] packed, int bits) {
    java.util.Arrays.fill(packed, (byte) 0);
    for (int i = 0; i < bins.length; i++) {
      int bitOffset = i * bits;
      int byteIdx = bitOffset >>> 3;
      int bitIdx = bitOffset & 7;
      packed[byteIdx] |= (byte) (bins[i] << bitIdx);
      if (bitIdx + bits > 8 && byteIdx + 1 < packed.length) {
        packed[byteIdx + 1] |= (byte) (bins[i] >>> (8 - bitIdx));
      }
    }
  }

  private static int unpackBin(byte[] packed, int index, int bits) {
    int bitOffset = index * bits;
    int byteIdx = bitOffset >>> 3;
    int bitIdx = bitOffset & 7;
    int mask = (1 << bits) - 1;
    int val = (packed[byteIdx] & 0xFF) >>> bitIdx;
    if (bitIdx + bits > 8 && byteIdx + 1 < packed.length) {
      val |= (packed[byteIdx + 1] & 0xFF) << (8 - bitIdx);
    }
    return val & mask;
  }
}
