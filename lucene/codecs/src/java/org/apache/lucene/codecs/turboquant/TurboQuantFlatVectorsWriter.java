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
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;

/**
 * Writes TurboQuant quantized vectors to disk. Raw float vectors are held in memory during indexing
 * for HNSW graph building but are NOT persisted — only quantized data is written.
 */
final class TurboQuantFlatVectorsWriter extends FlatVectorsWriter {

  private final TurboQuantFlatVectorsFormat format;
  private final IndexOutput dataOut;
  private final IndexOutput metaOut;
  private final org.apache.lucene.store.Directory directory;
  private final org.apache.lucene.store.IOContext ioContext;
  private final List<FieldWriter> fields = new ArrayList<>();
  private boolean finished;

  TurboQuantFlatVectorsWriter(SegmentWriteState state, TurboQuantFlatVectorsFormat format)
      throws IOException {
    super(new TurboQuantFlatScorer());
    this.format = format;
    this.directory = state.directory;
    this.ioContext = state.context;

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, format.metaExtension);
    String dataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, format.dataExtension);

    boolean success = false;
    try {
      metaOut = state.directory.createOutput(metaFileName, state.context);
      dataOut = state.directory.createOutput(dataFileName, state.context);
      CodecUtil.writeIndexHeader(
          metaOut,
          format.codecName,
          TurboQuantFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          dataOut,
          format.codecName,
          TurboQuantFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (!success) {
        close();
      }
    }
  }

  @Override
  public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    FieldWriter fw = new FieldWriter(fieldInfo);
    fields.add(fw);
    return fw;
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (FieldWriter fw : fields) {
      fw.flush(dataOut, metaOut, sortMap);
      fw.finish();
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      return;
    }
    finished = true;
    for (FieldWriter fw : fields) {
      fw.finish();
    }
    metaOut.writeInt(-1); // end-of-fields sentinel
    CodecUtil.writeFooter(metaOut);
    CodecUtil.writeFooter(dataOut);
  }

  /** Compute bytes per vector for the current format. */
  private int bytesPerVec(int dim) {
    return format.isExpanded() ? 4 + dim : TurboQuantVector.totalBytes(dim, format.bits);
  }

  /** Write field metadata to the meta output. */
  private void writeFieldMeta(IndexOutput metaOut, int fieldNumber, int dim, int numVecs,
      long dataOffset) throws IOException {
    metaOut.writeInt(fieldNumber);
    metaOut.writeInt(dim);
    metaOut.writeInt(numVecs);
    metaOut.writeLong(format.seed);
    metaOut.writeLong(dataOffset);
    metaOut.writeByte((byte) format.bits);
  }

  @Override
  public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(
      FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    int dim = fieldInfo.getVectorDimension();
    int bytesPerVec = bytesPerVec(dim);

    // Stream quantized vectors through a temp file to avoid heap pressure.
    // For TurboQuant source segments, byte-copy directly (lossless, fast).
    // For other codecs, re-encode from float32.
    // We use MergedVectorValues for correct doc ordering and liveDocs handling,
    // but check if the underlying FloatVectorValues is our StubVectorValues
    // to enable byte-copy.
    int numVecs = 0;
    byte[] buf = new byte[bytesPerVec];

    FloatVectorValues merged = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(
        fieldInfo, mergeState);

    // Detect if all sources are TurboQuant by checking each reader
    // Build a map from source reader index → StubVectorValues for byte-copy
    TurboQuantFlatVectorsReader.StubVectorValues[] tqSources =
        new TurboQuantFlatVectorsReader.StubVectorValues[mergeState.knnVectorsReaders.length];
    boolean allTQ = true;
    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      KnnVectorsReader kvr = mergeState.knnVectorsReaders[i];
      if (kvr == null) continue;
      FloatVectorValues fvv = kvr.getFloatVectorValues(fieldInfo.name);
      if (fvv == null) continue;
      if (fvv instanceof TurboQuantFlatVectorsReader.StubVectorValues stub) {
        tqSources[i] = stub;
      } else {
        allTQ = false;
        break;
      }
    }

    if (allTQ) {
      // Optimized byte-copy path: skip temp file entirely.
      // 1. Count total vectors and collect (mergedDocID, sourceIdx, sourceOrd) tuples
      // 2. Sort by mergedDocID for correct output order
      // 3. Write directly to dataOut + mergedData in one pass
      int totalVecs = 0;
      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        if (tqSources[i] == null) continue;
        FloatVectorValues fvv = mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldInfo.name);
        if (fvv == null) continue;
        var it = fvv.iterator();
        while (it.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
          if (mergeState.docMaps[i].get(it.docID()) != -1) totalVecs++;
        }
      }

      // Write metadata now that we know the count
      writeFieldMeta(metaOut, fieldInfo.number, dim, totalVecs, dataOut.getFilePointer());

      // Allocate scorer buffer
      Arena mergeArena = Arena.ofShared();
      MemorySegment mergedData = mergeArena.allocate((long) totalVecs * bytesPerVec);

      // Collect and sort entries
      int[][] entries = new int[totalVecs][3]; // [mergedDocID, sourceIdx, sourceOrd]
      int idx = 0;
      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        if (tqSources[i] == null) continue;
        FloatVectorValues fvv = mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldInfo.name);
        if (fvv == null) continue;
        var it = fvv.iterator();
        while (it.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
          int mappedDoc = mergeState.docMaps[i].get(it.docID());
          if (mappedDoc != -1) {
            entries[idx][0] = mappedDoc;
            entries[idx][1] = i;
            entries[idx][2] = it.index();
            idx++;
          }
        }
      }
      java.util.Arrays.sort(entries, (a, b) -> Integer.compare(a[0], b[0]));

      // Single-pass write: source MemorySegment → buf → dataOut + mergedData
      long outOffset = 0;
      for (int[] entry : entries) {
        MemorySegment.copy(
            tqSources[entry[1]].quantizedData, (long) entry[2] * bytesPerVec,
            MemorySegment.ofArray(buf), 0, bytesPerVec);
        dataOut.writeBytes(buf, bytesPerVec);
        MemorySegment.copy(MemorySegment.ofArray(buf), 0, mergedData, outOffset, bytesPerVec);
        outOffset += bytesPerVec;
      }
      numVecs = totalVecs;

      // Skip temp file path — go directly to scorer creation
      final int finalNumVecs = numVecs;
      return new CloseableRandomVectorScorerSupplier() {
        @Override
        public int totalVectorCount() {
          return finalNumVecs;
        }

        @Override
        public UpdateableRandomVectorScorer scorer() {
          return new TqMergePairScorer(
              mergedData, finalNumVecs, dim, bytesPerVec, format.bits);
        }

        @Override
        public RandomVectorScorerSupplier copy() {
          return this;
        }

        @Override
        public void close() throws IOException {
          mergeArena.close();
        }
      };
    } else {
      // Re-encode path: cross-codec merge or mixed sources
      IndexOutput tempOut = directory.createTempOutput(
          dataOut.getName(), "tq_merge", ioContext);
      TurboQuantEncoder encoder = new TurboQuantEncoder(dim, format.bits, format.seed);
      var iter = merged.iterator();
      while (iter.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
        float[] vec = merged.vectorValue(iter.index());
        try (Arena arena = Arena.ofConfined()) {
          TurboQuantVector tqv = encoder.encode(arena, vec);
          tqv.segment().asByteBuffer().get(buf, 0, bytesPerVec);
        }
        tempOut.writeBytes(buf, bytesPerVec);
        numVecs++;
      }

      CodecUtil.writeFooter(tempOut);
      IOUtils.close(tempOut);

      // Write metadata now that we know the actual count
      writeFieldMeta(metaOut, fieldInfo.number, dim, numVecs, dataOut.getFilePointer());

      // Copy temp file to final output and build off-heap scorer buffer
      Arena mergeArena = Arena.ofShared();
      MemorySegment mergedData = mergeArena.allocate((long) numVecs * bytesPerVec);
      IndexInput tempIn = directory.openInput(tempOut.getName(), ioContext);
      long remaining = (long) numVecs * bytesPerVec;
      long offset = 0;
      while (remaining > 0) {
        int toRead = (int) Math.min(buf.length, remaining);
        tempIn.readBytes(buf, 0, toRead);
        dataOut.writeBytes(buf, toRead);
        MemorySegment.copy(MemorySegment.ofArray(buf), 0, mergedData, offset, toRead);
        offset += toRead;
        remaining -= toRead;
      }
      IOUtils.close(tempIn);
      directory.deleteFile(tempOut.getName());

      final int finalNumVecs = numVecs;

      return new CloseableRandomVectorScorerSupplier() {
        @Override
        public int totalVectorCount() {
          return finalNumVecs;
        }

        @Override
        public UpdateableRandomVectorScorer scorer() {
          return new TqMergePairScorer(
              mergedData, finalNumVecs, dim, bytesPerVec, format.bits);
        }

        @Override
        public RandomVectorScorerSupplier copy() {
          return this;
        }

        @Override
        public void close() {
          mergeArena.close();
        }
      };
    } // end else (re-encode path)
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(metaOut, dataOut);
  }

  @Override
  public long ramBytesUsed() {
    long bytes = 0;
    for (FieldWriter fw : fields) {
      bytes += fw.ramBytesUsed();
    }
    return bytes;
  }

  /** Per-field writer: buffers raw vectors for graph building, writes quantized on flush. */
  private final class FieldWriter extends FlatFieldVectorsWriter<float[]> {
    private final FieldInfo fieldInfo;
    private final List<float[]> vectors = new ArrayList<>();
    private final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    private boolean fieldFinished;

    FieldWriter(FieldInfo fieldInfo) {
      this.fieldInfo = fieldInfo;
    }

    @Override
    public void addValue(int docID, float[] value) throws IOException {
      if (fieldFinished) {
        throw new IllegalStateException("already finished");
      }
      vectors.add(value.clone());
      docsWithField.add(docID);
    }

    @Override
    public float[] copyValue(float[] value) {
      return value.clone();
    }

    @Override
    public List<float[]> getVectors() {
      return vectors;
    }

    @Override
    public DocsWithFieldSet getDocsWithFieldSet() {
      return docsWithField;
    }

    @Override
    public void finish() throws IOException {
      fieldFinished = true;
    }

    @Override
    public boolean isFinished() {
      return fieldFinished;
    }

    @Override
    public long ramBytesUsed() {
      return vectors.isEmpty() ? 0 : (long) vectors.size() * vectors.get(0).length * 4;
    }

    void flush(IndexOutput dataOut, IndexOutput metaOut, Sorter.DocMap sortMap) throws IOException {
      int dim = fieldInfo.getVectorDimension();
      int numVecs = vectors.size();

      writeFieldMeta(metaOut, fieldInfo.number, dim, numVecs, dataOut.getFilePointer());

      if (numVecs == 0) {
        return;
      }

      TurboQuantEncoder encoder = new TurboQuantEncoder(dim, format.bits, format.seed);

      if (format.isExpanded()) {
        writeExpandedVectors(encoder, dim, numVecs, sortMap, dataOut);
      } else {
        writePackedVectors(encoder, dim, numVecs, sortMap, dataOut);
      }
    }

    private void writeExpandedVectors(
        TurboQuantEncoder encoder, int dim, int numVecs, Sorter.DocMap sortMap, IndexOutput dataOut)
        throws IOException {
      int bytesPerVec = 4 + dim;
      byte[] buf = new byte[bytesPerVec];
      float sigma = 1.0f / (float) Math.sqrt(dim);
      float[] centroids = PolarQuant.centroids(format.bits);
      float cMax = 0;
      for (float c : centroids) {
        cMax = Math.max(cMax, Math.abs(c * sigma));
      }
      float cScale = (cMax > 0) ? 127f / cMax : 1f;
      int levels = PolarQuant.levels(format.bits);
      byte[] centroidInt8 = new byte[levels];
      for (int j = 0; j < levels; j++) {
        centroidInt8[j] = (byte) Math.round(centroids[j] * sigma * cScale);
      }

      try (Arena arena = Arena.ofConfined()) {
        for (int i = 0; i < numVecs; i++) {
          int ord = (sortMap != null) ? sortMap.newToOld(i) : i;
          TurboQuantVector tqv = encoder.encode(arena, vectors.get(ord));
          float norm = tqv.getNorm();
          int normBits = Float.floatToIntBits(norm);
          buf[0] = (byte) normBits;
          buf[1] = (byte) (normBits >> 8);
          buf[2] = (byte) (normBits >> 16);
          buf[3] = (byte) (normBits >> 24);
          for (int d = 0; d < dim; d++) {
            buf[4 + d] = centroidInt8[tqv.getBin(d, format.bits) & 0xFF];
          }
          dataOut.writeBytes(buf, bytesPerVec);
        }
      }
    }

    private void writePackedVectors(
        TurboQuantEncoder encoder, int dim, int numVecs, Sorter.DocMap sortMap, IndexOutput dataOut)
        throws IOException {
      int bytesPerVec = TurboQuantVector.totalBytes(dim, format.bits);
      byte[] buf = new byte[bytesPerVec];
      try (Arena arena = Arena.ofConfined()) {
        for (int i = 0; i < numVecs; i++) {
          int ord = (sortMap != null) ? sortMap.newToOld(i) : i;
          TurboQuantVector tqv = encoder.encode(arena, vectors.get(ord));
          tqv.segment().asByteBuffer().get(buf, 0, bytesPerVec);
          dataOut.writeBytes(buf, bytesPerVec);
        }
      }
    }
  }

  /**
   * FlatVectorsScorer that uses raw float vectors for graph building (in memory) and TurboQuant for
   * query-time scoring.
   */
  static final class TurboQuantFlatScorer implements FlatVectorsScorer {

    TurboQuantFlatScorer() {}

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction sim, KnnVectorValues vectorValues) throws IOException {
      FloatVectorValues fvv = (FloatVectorValues) vectorValues;
      return new RandomVectorScorerSupplier() {
        @Override
        public UpdateableRandomVectorScorer scorer() throws IOException {
          FloatVectorValues copy = fvv.copy();
          return new UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer(copy) {
            private float[] currentVec;

            @Override
            public void setScoringOrdinal(int node) throws IOException {
              currentVec = copy.vectorValue(node);
            }

            @Override
            public float score(int node) throws IOException {
              return sim.compare(currentVec, copy.vectorValue(node));
            }
          };
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
          return this;
        }
      };
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction sim, KnnVectorValues vectorValues, float[] target)
        throws IOException {
      FloatVectorValues fvv = (FloatVectorValues) vectorValues;
      return new RandomVectorScorer.AbstractRandomVectorScorer(fvv) {
        @Override
        public float score(int node) throws IOException {
          return sim.compare(target, fvv.vectorValue(node));
        }
      };
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction sim, KnnVectorValues vectorValues, byte[] target)
        throws IOException {
      throw new UnsupportedOperationException("TurboQuant only supports float vectors");
    }
  }

  /**
   * Pair scorer for HNSW graph building during merge. Uses int8 dot product for 8-bit (expanded
   * layout), falls back to bit-width-specific scoring for packed modes.
   */
  private static final class TqMergePairScorer implements UpdateableRandomVectorScorer {
    private static final java.lang.foreign.ValueLayout.OfFloat FLOAT_LE =
        java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
    private static final java.lang.foreign.ValueLayout.OfInt INT_LE =
        java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED;

    private final MemorySegment data;
    private final int numVecs;
    private final int dim;
    private final int bytesPerVec;
    private final int packedBinsLen;
    private final byte[] currentInt8;
    private final byte[] currentDeinterleaved;
    private final byte[] centroidInt8;
    private final float int8InvScale;
    private final int bits;
    private final int intWordsPerVec;
    private final int[] currentBitWords;
    private float currentNorm;

    TqMergePairScorer(
        MemorySegment data, int numVecs, int dim, int bytesPerVec, int bits) {
      this.data = data;
      this.numVecs = numVecs;
      this.dim = dim;
      this.bytesPerVec = bytesPerVec;
      this.bits = bits;
      this.packedBinsLen = (bits == 8) ? 0 : TurboQuantVector.packedBinsBytes(dim, bits);
      this.currentInt8 = (bits == 8) ? new byte[dim] : null;
      if (bits == 1 || bits == 2) {
        this.intWordsPerVec = (packedBinsLen + 3) >>> 2;
        this.currentBitWords = new int[intWordsPerVec];
        this.int8InvScale = 0;
        this.centroidInt8 = null;
        this.currentDeinterleaved = null;
      } else if (bits == 4) {
        this.intWordsPerVec = 0;
        this.currentBitWords = null;
        float sigma = 1.0f / (float) Math.sqrt(dim);
        float[] centroids = PolarQuant.centroids(bits);
        float cMax = 0;
        for (float c : centroids) cMax = Math.max(cMax, Math.abs(c * sigma));
        float cScale = (cMax > 0) ? 127f / cMax : 1f;
        this.int8InvScale = 1f / (cScale * cScale * dim);
        this.centroidInt8 = new byte[centroids.length];
        for (int j = 0; j < centroids.length; j++) {
          centroidInt8[j] = (byte) Math.round(centroids[j] * sigma * cScale);
        }
        this.currentDeinterleaved = (bits == 4) ? new byte[dim] : null;
      } else {
        this.intWordsPerVec = 0;
        this.currentBitWords = null;
        this.int8InvScale = 0;
        this.centroidInt8 = null;
        this.currentDeinterleaved = null;
      }
    }

    @Override
    public int maxOrd() {
      return numVecs;
    }

    @Override
    public void setScoringOrdinal(int node) {
      long base = (long) node * bytesPerVec;
      currentNorm = data.get(FLOAT_LE, base);
      if (bits == 8) {
        MemorySegment.copy(data, base + 4, MemorySegment.ofArray(currentInt8), 0, dim);
      } else if (bits == 1 || bits == 2) {
        long off = base + 4;
        for (int w = 0; w < intWordsPerVec; w++) {
          currentBitWords[w] = data.get(INT_LE, off + (long) w * 4);
        }
      } else {
        for (int i = 0; i < packedBinsLen; i++) {
          int packed = data.get(java.lang.foreign.ValueLayout.JAVA_BYTE, base + 4 + i) & 0xFF;
          currentDeinterleaved[i] = centroidInt8[packed & 0x0F];
          currentDeinterleaved[packedBinsLen + i] = centroidInt8[packed >>> 4];
        }
      }
    }

    @Override
    public float score(int node) {
      long base = (long) node * bytesPerVec;
      float otherNorm = data.get(FLOAT_LE, base);
      if (currentNorm < 1e-30f || otherNorm < 1e-30f) {
        return 0f;
      }
      if (bits == 8) {
        int dot = TurboQuantScorer.dotProductInt8(currentInt8, data, base + 4, dim);
        return Math.max(0f, (1f + currentNorm * otherNorm * dot / (127f * 127f * dim)) / 2f);
      }
      if (bits == 1 || bits == 2) {
        long off = base + 4;
        int matchBits = 0;
        for (int w = 0; w < intWordsPerVec; w++) {
          int otherWord = data.get(INT_LE, off + (long) w * 4);
          matchBits += Integer.bitCount(~(currentBitWords[w] ^ otherWord));
        }
        float sim = (float) matchBits / (intWordsPerVec * 32);
        return (1f + currentNorm * otherNorm * (2f * sim - 1f)) / 2f;
      }
      // 4-bit: use float LUT path (same as search scorer)
      // Need rotatedQuery and centroidLUT — but merge scorer doesn't have them.
      // Fall back to int8 path for merge.
      int intDot =
          TurboQuantScorer.scorePolarInt8Direct(
              currentDeinterleaved, centroidInt8, data, base + 4, packedBinsLen);
      float rawDot = currentNorm * otherNorm * intDot * int8InvScale;
      return (1f + rawDot) / 2f;
    }
  }
}
