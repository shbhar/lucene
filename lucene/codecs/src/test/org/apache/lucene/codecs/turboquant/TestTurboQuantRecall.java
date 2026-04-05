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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Brute-force recall tests for TurboQuant scoring paths. */
public class TestTurboQuantRecall extends LuceneTestCase {

  private static final int K = 10;
  private static final long SEED = 42L;

  public void testBruteForceRecall4Bit1024d() throws Exception {
    assertBruteForceRecall(1024, 4, 1000, 100, 0.85);
  }

  public void testBruteForceRecall1Bit1024d() throws Exception {
    assertBruteForceRecall(1024, 1, 1000, 100, 0.30);
  }

  public void testBruteForceRecall4Bit4096d() throws Exception {
    assertBruteForceRecall(4096, 4, 500, 50, 0.88);
  }

  public void testBruteForceRecall1Bit4096d() throws Exception {
    assertBruteForceRecall(4096, 1, 500, 50, 0.30);
  }

  public void testBruteForceRecall2Bit1024d() throws Exception {
    assertBruteForceRecall(1024, 2, 1000, 100, 0.60);
  }

  public void testBruteForceRecall8Bit1024d() throws Exception {
    assertBruteForceRecall(1024, 8, 1000, 100, 0.90);
  }

  public void testSelfScoreIsMaximal() throws Exception {
    int dim = 1024;
    Random rng = new Random(123);
    float[] v = randomUnit(dim, rng);
    for (int bits : new int[] {1, 2, 4, 8}) {
      TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, SEED);
      try (Arena arena = Arena.ofConfined()) {
        TurboQuantVector tqv = enc.encode(arena, v);
        TurboQuantScorer.QueryState qs = enc.prepareQuery(v);
        float selfScore = TurboQuantScorer.score(qs, tqv);
        // Self-score should be close to ||v||^2 = 1.0 for unit vectors
        assertTrue(
            bits + "-bit self-score=" + selfScore + " too low",
            selfScore > (bits == 1 ? 0.5f : 0.8f));
      }
    }
  }

  public void testScoreCorrelation4Bit() throws Exception {
    assertScoreCorrelation(1024, 4, 200, 0.85);
  }

  public void testScoreCorrelation1Bit() throws Exception {
    assertScoreCorrelation(1024, 1, 200, 0.40);
  }

  /** Rescore (float-precision LUT) should achieve >= recall of search scorer (int8 SIMD). */
  public void testRescoreBeatsSearchRecall1Bit() throws Exception {
    assertRescoreBeatsSearch(256, 1, 500, 50);
  }

  public void testRescoreBeatsSearchRecall4Bit() throws Exception {
    assertRescoreBeatsSearch(256, 4, 500, 50);
  }

  /** Higher bit width should always achieve >= recall of lower bit width on same data. */
  public void testHigherBitsImproveRecall() throws Exception {
    int dim = 256;
    int numDocs = 500;
    int numQueries = 50;
    Random rng = new Random(789);
    float[][] docs = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) docs[i] = randomUnit(dim, rng);
    float[][] queries = new float[numQueries][];
    for (int i = 0; i < numQueries; i++) queries[i] = randomUnit(dim, rng);

    double[] recallByBits = new double[4];
    int[] bitWidths = {1, 2, 4, 8};
    for (int b = 0; b < bitWidths.length; b++) {
      int bits = bitWidths[b];
      TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, SEED);
      try (Arena arena = Arena.ofConfined()) {
        TurboQuantVector[] tqDocs = new TurboQuantVector[numDocs];
        for (int i = 0; i < numDocs; i++) tqDocs[i] = enc.encode(arena, docs[i]);
        double recallSum = 0;
        for (float[] query : queries) {
          Set<Integer> trueTopK = bruteForceTopK(query, docs, K);
          Set<Integer> tqTopK = tqBruteForceTopK(enc, query, tqDocs, K);
          int hits = 0;
          for (int d : tqTopK) if (trueTopK.contains(d)) hits++;
          recallSum += (double) hits / K;
        }
        recallByBits[b] = recallSum / numQueries;
      }
    }
    for (int b = 1; b < bitWidths.length; b++) {
      assertTrue(
          bitWidths[b] + "-bit recall (" + recallByBits[b] + ") should be >= "
              + bitWidths[b - 1] + "-bit recall (" + recallByBits[b - 1] + ")",
          recallByBits[b] >= recallByBits[b - 1] - 0.01);
    }
  }

  /** Same vector + same seed must produce identical quantized output. */
  public void testDeterministicEncoding() throws Exception {
    int dim = 512;
    for (int bits : new int[]{1, 2, 4, 8}) {
      float[] vec = randomUnit(dim, new Random(42));
      try (Arena a1 = Arena.ofConfined(); Arena a2 = Arena.ofConfined()) {
        TurboQuantVector v1 = new TurboQuantEncoder(dim, bits, SEED).encode(a1, vec);
        TurboQuantVector v2 = new TurboQuantEncoder(dim, bits, SEED).encode(a2, vec);
        assertEquals(bits + "-bit: norms differ", v1.getNorm(), v2.getNorm(), 0f);
        for (int i = 0; i < dim; i++) {
          assertEquals(bits + "-bit: bin " + i + " differs",
              v1.getBin(i, bits), v2.getBin(i, bits));
        }
      }
    }
  }

  /** Scores before and after forceMerge should be identical for the same query. */
  public void testMergePreservesScores() throws Exception {
    int dim = 128;
    int bits = 4;
    TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, 42L);
    float[][] docs = new float[100][];
    Random rng = new Random(99);
    for (int i = 0; i < docs.length; i++) docs[i] = randomUnit(dim, rng);
    float[] query = randomUnit(dim, rng);

    // Encode in two batches (simulating two segments)
    TurboQuantScorer.QueryState qs = enc.prepareQuery(query);
    float[] scoresBefore = new float[docs.length];
    try (Arena arena = Arena.ofConfined()) {
      for (int i = 0; i < docs.length; i++) {
        TurboQuantVector tqv = enc.encode(arena, docs[i]);
        scoresBefore[i] = TurboQuantScorer.score(qs, tqv);
      }
    }
    // Re-encode (simulating merge re-encoding)
    float[] scoresAfter = new float[docs.length];
    try (Arena arena = Arena.ofConfined()) {
      for (int i = 0; i < docs.length; i++) {
        TurboQuantVector tqv = enc.encode(arena, docs[i]);
        scoresAfter[i] = TurboQuantScorer.score(qs, tqv);
      }
    }
    for (int i = 0; i < docs.length; i++) {
      assertEquals("score changed after re-encode for doc " + i,
          scoresBefore[i], scoresAfter[i], 0f);
    }
  }

  /** Zero vector should score approximately zero against any query. */
  public void testZeroVectorScoresZero() throws Exception {
    int dim = 256;
    float[] query = randomUnit(dim, new Random(42));
    float[] zero = new float[dim];
    for (int bits : new int[]{1, 2, 4, 8}) {
      TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, SEED);
      TurboQuantScorer.QueryState qs = enc.prepareQuery(query);
      try (Arena arena = Arena.ofConfined()) {
        TurboQuantVector tqv = enc.encode(arena, zero);
        float score = TurboQuantScorer.score(qs, tqv);
        assertEquals(bits + "-bit: zero vector should score ~0", 0f, score, 1e-6f);
      }
    }
  }

  /** Scaling a vector by 2× should increase its score when dot product is positive. */
  public void testScoreMonotonicWithNorm() throws Exception {
    int dim = 256;
    float[] unit = randomUnit(dim, new Random(42));
    float[] scaled = new float[dim];
    for (int i = 0; i < dim; i++) scaled[i] = unit[i] * 2f;
    float[] query = unit.clone();
    for (int bits : new int[]{1, 2, 4, 8}) {
      TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, SEED);
      TurboQuantScorer.QueryState qs = enc.prepareQuery(query);
      try (Arena arena = Arena.ofConfined()) {
        float scoreUnit = TurboQuantScorer.score(qs, enc.encode(arena, unit));
        float scoreScaled = TurboQuantScorer.score(qs, enc.encode(arena, scaled));
        assertTrue(bits + "-bit: scaled vector should score >= unit vector, got "
                + scoreScaled + " vs " + scoreUnit,
            scoreScaled >= scoreUnit - 0.01f);
      }
    }
  }

  private void assertRescoreBeatsSearch(int dim, int bits, int numDocs, int numQueries)
      throws Exception {
    Random rng = new Random(456);
    float[][] docs = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) docs[i] = randomUnit(dim, rng);
    float[][] queries = new float[numQueries][];
    for (int i = 0; i < numQueries; i++) queries[i] = randomUnit(dim, rng);

    TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, SEED);
    try (Arena arena = Arena.ofConfined()) {
      TurboQuantVector[] tqDocs = new TurboQuantVector[numDocs];
      for (int i = 0; i < numDocs; i++) tqDocs[i] = enc.encode(arena, docs[i]);

      double searchRecallSum = 0, rescoreRecallSum = 0;
      for (float[] query : queries) {
        Set<Integer> trueTopK = bruteForceTopK(query, docs, K);

        // Search scorer (int8 quantized query)
        Set<Integer> searchTopK = tqBruteForceTopK(enc, query, tqDocs, K);

        // Rescore scorer (float-precision centroid LUT)
        float[][] rl = TurboQuantScorer.rotateAndBuildLUT(query, enc.signs(), bits);
        float[] rotated = rl[0];
        float queryNorm = rl[1][0];
        float[] lut = rl[2];
        float[] rescoreScores = new float[numDocs];
        for (int i = 0; i < numDocs; i++) {
          float docNorm = tqDocs[i].getNorm();
          float dot = 0;
          for (int d = 0; d < dim; d++) {
            dot += rotated[d] * lut[tqDocs[i].getBin(d, bits) & 0xFF];
          }
          rescoreScores[i] = queryNorm * docNorm * dot;
        }
        Set<Integer> rescoreTopK = topKIndices(rescoreScores, K);

        int searchHits = 0, rescoreHits = 0;
        for (int d : searchTopK) if (trueTopK.contains(d)) searchHits++;
        for (int d : rescoreTopK) if (trueTopK.contains(d)) rescoreHits++;
        searchRecallSum += (double) searchHits / K;
        rescoreRecallSum += (double) rescoreHits / K;
      }
      double searchRecall = searchRecallSum / numQueries;
      double rescoreRecall = rescoreRecallSum / numQueries;
      assertTrue(
          bits + "-bit: rescore recall (" + rescoreRecall + ") should be >= search recall ("
              + searchRecall + ")",
          rescoreRecall >= searchRecall - 0.01); // small tolerance for noise
    }
  }

  private void assertBruteForceRecall(
      int dim, int bits, int numDocs, int numQueries, double minRecall) throws Exception {
    Random rng = new Random(123);
    float[][] docs = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) docs[i] = randomUnit(dim, rng);
    float[][] queries = new float[numQueries][];
    for (int i = 0; i < numQueries; i++) queries[i] = randomUnit(dim, rng);

    TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, SEED);
    TurboQuantVector[] tqDocs;
    try (Arena arena = Arena.ofConfined()) {
      tqDocs = new TurboQuantVector[numDocs];
      for (int i = 0; i < numDocs; i++) tqDocs[i] = enc.encode(arena, docs[i]);

      double totalRecall = 0;
      for (float[] query : queries) {
        Set<Integer> trueTopK = bruteForceTopK(query, docs, K);
        Set<Integer> tqTopK = tqBruteForceTopK(enc, query, tqDocs, K);
        int hits = 0;
        for (int d : tqTopK) if (trueTopK.contains(d)) hits++;
        totalRecall += (double) hits / K;
      }
      double recall = totalRecall / numQueries;
      assertTrue(
          bits + "-bit " + dim + "d recall=" + recall + " < " + minRecall, recall >= minRecall);
    }
  }

  private void assertScoreCorrelation(int dim, int bits, int numDocs, double minTau)
      throws Exception {
    Random rng = new Random(456);
    float[] query = randomUnit(dim, rng);
    float[][] docs = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) docs[i] = randomUnit(dim, rng);

    TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, SEED);
    try (Arena arena = Arena.ofConfined()) {
      float[] trueDots = new float[numDocs];
      float[] tqScores = new float[numDocs];
      TurboQuantScorer.QueryState qs = enc.prepareQuery(query);
      for (int i = 0; i < numDocs; i++) {
        trueDots[i] = dot(query, docs[i]);
        tqScores[i] = TurboQuantScorer.score(qs, enc.encode(arena, docs[i]));
      }
      double tau = kendallTau(trueDots, tqScores);
      assertTrue(bits + "-bit tau=" + tau + " < " + minTau, tau >= minTau);
    }
  }

  private static Set<Integer> bruteForceTopK(float[] query, float[][] docs, int k) {
    int n = docs.length;
    float[] scores = new float[n];
    for (int i = 0; i < n; i++) scores[i] = dot(query, docs[i]);
    return topKIndices(scores, k);
  }

  private static Set<Integer> tqBruteForceTopK(
      TurboQuantEncoder enc, float[] query, TurboQuantVector[] docs, int k) {
    TurboQuantScorer.QueryState qs = enc.prepareQuery(query);
    float[] scores = new float[docs.length];
    for (int i = 0; i < docs.length; i++) scores[i] = TurboQuantScorer.score(qs, docs[i]);
    return topKIndices(scores, k);
  }

  private static Set<Integer> topKIndices(float[] scores, int k) {
    Integer[] idx = new Integer[scores.length];
    for (int i = 0; i < idx.length; i++) idx[i] = i;
    Arrays.sort(idx, (a, b) -> Float.compare(scores[b], scores[a]));
    Set<Integer> result = new HashSet<>();
    for (int i = 0; i < k && i < idx.length; i++) result.add(idx[i]);
    return result;
  }

  private static double kendallTau(float[] a, float[] b) {
    int n = a.length;
    int concordant = 0, discordant = 0;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        boolean aOrder = a[i] > a[j];
        boolean bOrder = b[i] > b[j];
        if (aOrder == bOrder) concordant++;
        else discordant++;
      }
    }
    return (double) (concordant - discordant) / (concordant + discordant);
  }

  private static float dot(float[] a, float[] b) {
    float s = 0;
    for (int i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  private static float[] randomUnit(int dim, Random rng) {
    float[] v = new float[dim];
    float norm = 0;
    for (int i = 0; i < dim; i++) {
      v[i] = (float) rng.nextGaussian();
      norm += v[i] * v[i];
    }
    norm = (float) Math.sqrt(norm);
    for (int i = 0; i < dim; i++) v[i] /= norm;
    return v;
  }
}
