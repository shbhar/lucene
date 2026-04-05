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

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;

/** End-to-end codec test: index → forceMerge → search → check recall. */
public class TestTurboQuantCodecRecall extends LuceneTestCase {

  private static final int K = 10;
  private static final String FIELD = "vec";

  public void testCodecRecall4Bit() throws Exception {
    assertCodecRecall(1024, 4, 1000, 50, 0.45);
  }

  public void testCodecRecall1Bit() throws Exception {
    assertCodecRecall(1024, 1, 1000, 50, 0.20);
  }

  private void assertCodecRecall(int dim, int bits, int numDocs, int numQueries, double minRecall)
      throws Exception {
    Random rng = new Random(42);
    float[][] docs = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) docs[i] = randomUnit(dim, rng);
    float[][] queries = new float[numQueries][];
    for (int i = 0; i < numQueries; i++) queries[i] = randomUnit(dim, rng);

    // Compute brute-force ground truth
    @SuppressWarnings("unchecked")
    Set<Integer>[] gt = (Set<Integer>[]) new Set<?>[numQueries];
    for (int q = 0; q < numQueries; q++) {
      float[] scores = new float[numDocs];
      for (int d = 0; d < numDocs; d++) scores[d] = dot(queries[q], docs[d]);
      gt[q] = topKIndices(scores, K);
    }

    // Index with TurboQuant codec
    Codec codec = createCodec(bits);
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig();
      iwc.setCodec(codec);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (float[] doc : docs) {
          Document d = new Document();
          d.add(new KnnFloatVectorField(FIELD, doc, VectorSimilarityFunction.DOT_PRODUCT));
          w.addDocument(d);
        }
        w.forceMerge(1);
      }

      // Search
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        double totalRecall = 0;
        for (int q = 0; q < numQueries; q++) {
          TopDocs td = searcher.search(new KnnFloatVectorQuery(FIELD, queries[q], K), K);
          Set<Integer> retrieved = new HashSet<>();
          for (ScoreDoc sd : td.scoreDocs) retrieved.add(sd.doc);
          int hits = 0;
          for (int d : retrieved) if (gt[q].contains(d)) hits++;
          totalRecall += (double) hits / K;
        }
        double recall = totalRecall / numQueries;
        assertTrue(bits + "-bit codec recall=" + recall + " < " + minRecall, recall >= minRecall);
      }
    }
  }

  private static Codec createCodec(int bits) {
    TurboQuantVectorsFormat tqFormat = new TurboQuantVectorsFormat(16, 100, bits);
    return new Lucene104Codec() {
      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        return tqFormat;
      }
    };
  }

  private static Set<Integer> topKIndices(float[] scores, int k) {
    Integer[] idx = new Integer[scores.length];
    for (int i = 0; i < idx.length; i++) idx[i] = i;
    java.util.Arrays.sort(idx, (a, b) -> Float.compare(scores[b], scores[a]));
    Set<Integer> result = new HashSet<>();
    for (int i = 0; i < k && i < idx.length; i++) result.add(idx[i]);
    return result;
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
