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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Off-heap packed representation of a TurboQuant-compressed vector.
 *
 * <p>Layout: {@code [float32 norm][byte[] packedBins]}
 *
 * <p>Bin packing depends on bit depth:
 *
 * <ul>
 *   <li>1-bit: 8 bins per byte
 *   <li>2-bit: 4 bins per byte
 *   <li>4-bit: 2 bins per byte (low nibble = even, high nibble = odd)
 *   <li>8-bit: 1 bin per byte
 * </ul>
 *
 * @lucene.experimental
 */
final class TurboQuantVector {

  private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED;
  private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;

  private static final long NORM_OFFSET = 0;
  static final long PACKED_BINS_OFFSET = 4;

  private final MemorySegment seg;
  private final int dim;
  private final int bits;

  private TurboQuantVector(MemorySegment seg, int dim, int bits) {
    this.seg = seg;
    this.dim = dim;
    this.bits = bits;
  }

  static int packedBinsBytes(int dim, int bits) {
    return (dim * bits + 7) >>> 3;
  }

  static long byteSize(int dim, int bits) {
    return 4L + packedBinsBytes(dim, bits);
  }

  static int totalBytes(int dim, int bits) {
    return (int) byteSize(dim, bits);
  }

  static TurboQuantVector allocate(Arena arena, int dim, int bits) {
    MemorySegment seg = arena.allocate(byteSize(dim, bits), 8);
    return new TurboQuantVector(seg, dim, bits);
  }

  static TurboQuantVector wrap(MemorySegment seg, int dim, int bits) {
    return new TurboQuantVector(seg, dim, bits);
  }

  void setNorm(float norm) {
    seg.set(FLOAT_LE, NORM_OFFSET, norm);
  }

  float getNorm() {
    return seg.get(FLOAT_LE, NORM_OFFSET);
  }

  void setBin(int i, byte bin, int bits) {
    switch (bits) {
      case 1 -> setBit(i, bin);
      case 2 -> set2Bit(i, bin);
      case 4 -> set4Bit(i, bin);
      case 8 -> set8Bit(i, bin);
      default -> throw new IllegalArgumentException("Unsupported bits: " + bits);
    }
  }

  byte getBin(int i, int bits) {
    return switch (bits) {
      case 1 -> getBit(i);
      case 2 -> get2Bit(i);
      case 4 -> get4Bit(i);
      case 8 -> get8Bit(i);
      default -> throw new IllegalArgumentException("Unsupported bits: " + bits);
    };
  }

  // --- 1-bit: 8 bins per byte ---

  private void setBit(int i, byte bin) {
    int byteIdx = i >>> 3;
    int bitIdx = i & 7;
    long offset = PACKED_BINS_OFFSET + byteIdx;
    byte existing = seg.get(BYTE_LE, offset);
    if ((bin & 1) == 1) {
      seg.set(BYTE_LE, offset, (byte) (existing | (1 << bitIdx)));
    } else {
      seg.set(BYTE_LE, offset, (byte) (existing & ~(1 << bitIdx)));
    }
  }

  private byte getBit(int i) {
    int byteIdx = i >>> 3;
    int bitIdx = i & 7;
    return (byte) ((seg.get(BYTE_LE, PACKED_BINS_OFFSET + byteIdx) >>> bitIdx) & 1);
  }

  // --- 2-bit: 4 bins per byte ---

  private void set2Bit(int i, byte bin) {
    int byteIdx = i >>> 2;
    int shift = (i & 3) * 2;
    long offset = PACKED_BINS_OFFSET + byteIdx;
    byte existing = seg.get(BYTE_LE, offset);
    existing = (byte) (existing & ~(0x03 << shift));
    seg.set(BYTE_LE, offset, (byte) (existing | ((bin & 0x03) << shift)));
  }

  private byte get2Bit(int i) {
    int byteIdx = i >>> 2;
    int shift = (i & 3) * 2;
    return (byte) ((seg.get(BYTE_LE, PACKED_BINS_OFFSET + byteIdx) >>> shift) & 0x03);
  }

  // --- 4-bit: 2 bins per byte (low nibble = even, high nibble = odd) ---

  private void set4Bit(int i, byte bin) {
    int byteIdx = i >>> 1;
    long offset = PACKED_BINS_OFFSET + byteIdx;
    byte existing = seg.get(BYTE_LE, offset);
    if ((i & 1) == 0) {
      seg.set(BYTE_LE, offset, (byte) ((existing & 0xF0) | (bin & 0x0F)));
    } else {
      seg.set(BYTE_LE, offset, (byte) ((existing & 0x0F) | ((bin & 0x0F) << 4)));
    }
  }

  private byte get4Bit(int i) {
    int byteIdx = i >>> 1;
    byte packed = seg.get(BYTE_LE, PACKED_BINS_OFFSET + byteIdx);
    return (byte) (((i & 1) == 0) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F));
  }

  // --- 8-bit: 1 bin per byte ---

  private void set8Bit(int i, byte bin) {
    seg.set(BYTE_LE, PACKED_BINS_OFFSET + i, bin);
  }

  private byte get8Bit(int i) {
    return seg.get(BYTE_LE, PACKED_BINS_OFFSET + i);
  }

  byte getPackedByte(int byteIdx) {
    return seg.get(BYTE_LE, PACKED_BINS_OFFSET + byteIdx);
  }

  // --- Layout info ---

  MemorySegment segment() {
    return seg;
  }

  int dim() {
    return dim;
  }

  int bits() {
    return bits;
  }
}
