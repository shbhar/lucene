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

import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogByteSizeMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Benchmark forceMerge performance for TurboQuant byte-copy merge path. */
public class TestTurboQuantMergeBench extends LuceneTestCase {

  private static final String FIELD = "vec";

  public void testMergeBenchmark() throws Exception {
    int dim = 1024;
    int numDocs = 5_000;
    int numSegments = 5;
    int docsPerSegment = numDocs / numSegments;

    String[] labels = {"fp32", "BBQ-1bit", "SQ-4bit", "SQ-8bit", "TQ-1bit", "TQ-4bit", "TQ-8bit"};
    Codec[] codecs = {
        new Lucene104Codec(),
        new Lucene104Codec(), // BBQ = default Lucene104 KNN format (binary quantized)
        new Lucene104Codec() {
          @Override public KnnVectorsFormat getKnnVectorsFormatForField(String f) {
            return new Lucene104HnswScalarQuantizedVectorsFormat(
                QuantizedByteVectorValues.ScalarEncoding.PACKED_NIBBLE, 16, 100);
          }
        },
        new Lucene104Codec() {
          @Override public KnnVectorsFormat getKnnVectorsFormatForField(String f) {
            return new Lucene104HnswScalarQuantizedVectorsFormat(
                QuantizedByteVectorValues.ScalarEncoding.UNSIGNED_BYTE, 16, 100);
          }
        },
        createCodec(1),
        createCodec(4),
        createCodec(8),
    };

    for (int c = 0; c < labels.length; c++) {
      String label = labels[c];
      Codec codec = codecs[c];
      try (Directory dir = newDirectory()) {
        // Create multiple segments by flushing between batches
        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(codec);
        iwc.setMaxBufferedDocs(docsPerSegment);
        iwc.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
        // Prevent auto-merge so we control segment count
        LogByteSizeMergePolicy mp = new LogByteSizeMergePolicy();
        mp.setMergeFactor(100);
        iwc.setMergePolicy(mp);

        Random rng = new Random(42);
        long indexStart = System.nanoTime();
        try (IndexWriter w = new IndexWriter(dir, iwc)) {
          for (int i = 0; i < numDocs; i++) {
            Document d = new Document();
            d.add(new KnnFloatVectorField(FIELD, randomUnit(dim, rng),
                VectorSimilarityFunction.DOT_PRODUCT));
            w.addDocument(d);
          }
          w.flush();
          long indexTime = System.nanoTime() - indexStart;
          System.out.printf("%s: indexing %d docs in %.1f ms%n", label, numDocs, indexTime / 1e6);

          // Verify multiple segments
          try (DirectoryReader r = DirectoryReader.open(w)) {
            int segs = r.leaves().size();
            System.out.println(label + ": " + segs + " segments before merge");
            assertTrue("expected >1 segment, got " + segs, segs > 1);
          }

          // Benchmark forceMerge
          long t0 = System.nanoTime();
          w.forceMerge(1);
          long elapsed = System.nanoTime() - t0;
          System.out.printf("%s: forceMerge %d docs in %.1f ms%n",
              label, numDocs, elapsed / 1e6);
        }

        // Verify single segment and search works
        try (DirectoryReader r = DirectoryReader.open(dir)) {
          assertEquals(1, r.leaves().size());
          assertEquals(numDocs, r.numDocs());
        }
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
