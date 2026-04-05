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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * TurboQuant HNSW vector format using Lloyd-Max polar quantization with FWHT rotation.
 *
 * <p>Stores quantized vectors in .tqv/.tqm files alongside a standard Lucene99 HNSW graph. No raw
 * float32 vectors are stored, achieving true compression (unlike SQ/BBQ which store raw vectors for
 * rescore). Supports 1, 2, 4, and 8-bit quantization.
 *
 * <p>For rescore, use Lucene's built-in float32 rescore path (same as BBQ's {@code -overSample}
 * with {@code -rescore} in KnnGraphTester).
 */
public final class TurboQuantVectorsFormat extends KnnVectorsFormat {

  private final TurboQuantFlatVectorsFormat flatFormat;
  private final int maxConn;
  private final int beamWidth;

  /** Default: 4-bit, M=16, beamWidth=100. */
  public TurboQuantVectorsFormat() {
    this(16, 100, 4);
  }

  /**
   * @param maxConn HNSW max connections per node
   * @param beamWidth HNSW beam width during construction
   * @param bits quantization bits (1, 2, 4, or 8)
   */
  public TurboQuantVectorsFormat(int maxConn, int beamWidth, int bits) {
    super("TurboQuant");
    this.flatFormat = new TurboQuantFlatVectorsFormat(bits, 42L);
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Lucene99HnswVectorsWriter(
        state, maxConn, beamWidth, flatFormat.fieldsWriter(state), 0, null);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Lucene99HnswVectorsReader(state, flatFormat.fieldsReader(state));
  }

  /** FWHT rotation requires power-of-2 dimensions; 65536 = 2^16 is the practical upper bound. */
  @Override
  public int getMaxDimensions(String fieldName) {
    return 65536;
  }

  @Override
  public String toString() {
    return "TurboQuantVectorsFormat(maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", bits="
        + flatFormat.bits
        + ")";
  }
}
