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
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * TurboQuant as a {@link FlatVectorsFormat} — stores only quantized vectors on disk.
 *
 * <p>No raw float32 vectors are persisted. During indexing, raw float vectors are held in memory
 * for HNSW graph building. On flush, only quantized vectors are written to disk. At search time,
 * scoring uses quantized vectors via int8 SIMD paths.
 *
 * <p>Storage layout is auto-determined by bit depth: 8-bit uses expanded layout (1 byte/dim),
 * ≤4-bit uses packed layout for compression (e.g. 1-bit = 8 dims/byte → 30× compression).
 */
public final class TurboQuantFlatVectorsFormat extends FlatVectorsFormat {

  static final String TQV_EXTENSION = "tqv";
  static final String TQM_EXTENSION = "tqm";
  static final String TQV_CODEC = "TurboQuantFlat";
  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = 3;

  final int bits;
  final long seed;
  final String dataExtension;
  final String metaExtension;
  final String codecName;

  /** Default: 4-bit packed. */
  public TurboQuantFlatVectorsFormat() {
    this(4, 42L);
  }

  /**
   * Public constructor.
   *
   * @param bits quantization bits (1, 2, 4, or 8)
   * @param seed random seed for FWHT rotation
   */
  public TurboQuantFlatVectorsFormat(int bits, long seed) {
    this(bits, seed, TQV_EXTENSION, TQM_EXTENSION, TQV_CODEC);
  }

  /** Package-private constructor with custom file extensions for rescore format reuse. */
  TurboQuantFlatVectorsFormat(
      int bits, long seed, String dataExtension, String metaExtension, String codecName) {
    super(codecName);
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) {
      throw new IllegalArgumentException("bits must be 1, 2, 4, or 8, got " + bits);
    }
    this.bits = bits;
    this.seed = seed;
    this.dataExtension = dataExtension;
    this.metaExtension = metaExtension;
    this.codecName = codecName;
  }

  /** Whether this bit depth uses expanded (1 byte/dim) layout. 8-bit only. */
  boolean isExpanded() {
    return bits == 8;
  }

  @Override
  public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new TurboQuantFlatVectorsWriter(state, this);
  }

  @Override
  public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new TurboQuantFlatVectorsReader(state, this);
  }

  /** FWHT rotation requires power-of-2 dimensions; 65536 = 2^16 is the practical upper bound. */
  @Override
  public int getMaxDimensions(String fieldName) {
    return 65536;
  }
}
