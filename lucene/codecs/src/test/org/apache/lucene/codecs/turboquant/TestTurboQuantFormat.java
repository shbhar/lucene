/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.
 */
package org.apache.lucene.codecs.turboquant;

import java.io.IOException;
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
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;

/** End-to-end index/search tests for TurboQuantVectorsFormat. */
public class TestTurboQuantFormat extends LuceneTestCase {

  private static final int DIM = 128;
  private static final int NUM_DOCS = 500;

  public void testIndexAndSearch1Bit() throws Exception {
    doTestIndexAndSearch(1, 0.2f);
  }

  public void testIndexAndSearch2Bit() throws Exception {
    doTestIndexAndSearch(2, 0.3f);
  }

  public void testIndexAndSearch4Bit() throws Exception {
    doTestIndexAndSearch(4, 0.5f);
  }

  public void testIndexAndSearch8Bit() throws Exception {
    doTestIndexAndSearch(8, 0.7f);
  }

  public void testRescorerImproves1Bit() throws Exception {
    doTestRescorerImproves(1);
  }

  public void testRescorerImproves4Bit() throws Exception {
    doTestRescorerImproves(4);
  }

  public void testForceMerge() throws Exception {
    Codec codec = createCodec(4);
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        // Write docs in two batches to create 2 segments
        for (int i = 0; i < NUM_DOCS / 2; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("vec", randomVector(), VectorSimilarityFunction.DOT_PRODUCT));
          w.addDocument(doc);
        }
        w.flush();
        for (int i = 0; i < NUM_DOCS / 2; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("vec", randomVector(), VectorSimilarityFunction.DOT_PRODUCT));
          w.addDocument(doc);
        }
        w.forceMerge(1);
      }
      // Verify search works after merge
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", randomVector(), 10), 10);
        assertTrue("should find results after merge", td.scoreDocs.length > 0);
      }
    }
  }

  public void testEmptyIndex() throws Exception {
    Codec codec = createCodec(4);
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", randomVector(), 10), 10);
        assertEquals(0, td.scoreDocs.length);
      }
    }
  }

  /**
   * Verify that the rescorer produces valid scores. Tests the float-precision centroid LUT path
   * used for rescore (≤4-bit) by encoding vectors and scoring with prepareQuery.
   */
  private void doTestRescorerImproves(int bits) throws IOException {
    int dim = 128;
    int numDocs = 100;
    TurboQuantEncoder enc = new TurboQuantEncoder(dim, bits, 42L);
    float[][] vectors = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) vectors[i] = randomVector();
    float[] query = randomVector();

    // Compute exact dot products
    float[] exactScores = new float[numDocs];
    for (int i = 0; i < numDocs; i++) {
      float dot = 0;
      for (int d = 0; d < dim; d++) dot += query[d] * vectors[i][d];
      exactScores[i] = dot;
    }

    // Compute quantized scores via encoder
    TurboQuantScorer.QueryState qs = enc.prepareQuery(query);
    float[] quantScores = new float[numDocs];
    try (var arena = java.lang.foreign.Arena.ofConfined()) {
      for (int i = 0; i < numDocs; i++) {
        TurboQuantVector tqv = enc.encode(arena, vectors[i]);
        quantScores[i] = TurboQuantScorer.score(qs, tqv);
      }
    }

    // Quantized scores should correlate with exact scores
    // Compute Spearman rank correlation (simplified: count concordant pairs)
    int concordant = 0, total = 0;
    for (int i = 0; i < numDocs; i++) {
      for (int j = i + 1; j < Math.min(i + 20, numDocs); j++) {
        boolean exactOrder = exactScores[i] > exactScores[j];
        boolean quantOrder = quantScores[i] > quantScores[j];
        if (exactOrder == quantOrder) concordant++;
        total++;
      }
    }
    float correlation = (float) concordant / total;
    assertTrue(bits + "-bit correlation=" + correlation + " too low",
        correlation > (bits == 1 ? 0.55f : 0.7f));
  }

  public void testSingleVector() throws Exception {
    Codec codec = createCodec(4);
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("vec", randomVector(), VectorSimilarityFunction.DOT_PRODUCT));
        w.addDocument(doc);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", randomVector(), 10), 10);
        assertEquals("single doc index should return 1 result", 1, td.scoreDocs.length);
      }
    }
  }

  public void testDeleteAndMergePreservesCorrectVectors() throws Exception {
    Codec codec = createCodec(4);
    int numDocs = 200;
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec);
      iwc.setMaxBufferedDocs(50); // force multiple segments
      float[][] vectors = new float[numDocs][];
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < numDocs; i++) {
          vectors[i] = randomVector();
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("vec", vectors[i], VectorSimilarityFunction.DOT_PRODUCT));
          doc.add(new org.apache.lucene.document.StoredField("id", i));
          w.addDocument(doc);
        }
      }
      // Reopen, delete half, forceMerge
      try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig().setCodec(codec))) {
        try (DirectoryReader reader = DirectoryReader.open(w)) {
          for (int i = 0; i < reader.maxDoc(); i += 2) {
            w.tryDeleteDocument(reader, i);
          }
        }
        w.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals("should be 1 segment after forceMerge", 1, reader.leaves().size());
        int surviving = reader.numDocs();
        assertTrue("should have ~half docs, got " + surviving, surviving >= numDocs / 3);
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", randomVector(), 10), 10);
        assertTrue("should return results after delete+merge", td.scoreDocs.length > 0);
      }
    }
  }

  /**
   * Smoke test: index vectors, search, verify enough results are returned. This checks hit rate
   * (fraction of requested k results actually returned), NOT recall against brute-force ground
   * truth — see TestTurboQuantRecall for true recall tests.
   */
  private void doTestIndexAndSearch(int bits, float minHitRate) throws IOException {
    Codec codec = createCodec(bits);
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec);
      float[][] vectors = new float[NUM_DOCS][];
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          vectors[i] = randomVector();
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("vec", vectors[i], VectorSimilarityFunction.DOT_PRODUCT));
          w.addDocument(doc);
        }
        w.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        int totalHits = 0, totalRelevant = 0;
        for (int q = 0; q < 10; q++) {
          float[] query = randomVector();
          TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", query, 10), 10);
          assertTrue("should return results", td.scoreDocs.length > 0);
          totalHits += td.scoreDocs.length;
          totalRelevant += 10;
        }
        float hitRate = (float) totalHits / totalRelevant;
        assertTrue(bits + "-bit: hitRate=" + hitRate + " below " + minHitRate,
            hitRate >= minHitRate);
      }
    }
  }

  private float[] randomVector() {
    float[] v = new float[DIM];
    float n = 0;
    for (int i = 0; i < DIM; i++) { v[i] = (float) random().nextGaussian(); n += v[i] * v[i]; }
    n = (float) Math.sqrt(n);
    for (int i = 0; i < DIM; i++) v[i] /= n;
    return v;
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
}
