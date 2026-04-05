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

/**
 * TurboQuant vector quantization codec for HNSW vector search.
 *
 * <p>TurboQuant is a vector quantization codec based on the TurboQuant algorithm (Zandieh et al.,
 * ICLR 2026, <a href="https://arxiv.org/abs/2504.19874">arXiv:2504.19874</a>). The algorithm
 * achieves near-optimal distortion (within 2.7× of the information-theoretic lower bound) by:
 *
 * <ol>
 *   <li><b>Random rotation</b> — rotates vectors so each coordinate follows a concentrated Beta
 *       distribution (≈ Gaussian in high dimensions). This codec uses FWHT (Fast Walsh-Hadamard
 *       Transform) from the FJLT literature (Ailon &amp; Chazelle, STOC 2006) instead of the
 *       paper's QR rotation, achieving O(d log d) instead of O(d²) with equivalent quality.
 *   <li><b>Lloyd-Max polar quantization</b> — applies precomputed optimal scalar quantizers per
 *       coordinate. No per-vector or per-dimension calibration needed.
 * </ol>
 *
 * <p>Supported configurations:
 *
 * <ul>
 *   <li>1-bit: binary Lloyd-Max polar quantization (highest compression, needs rescore)
 *   <li>2-bit: 4-level Lloyd-Max polar quantization
 *   <li>4-bit: 16-level Lloyd-Max polar quantization (packed 2 bins/byte or expanded 1 byte/dim)
 *   <li>8-bit: 256-level Lloyd-Max polar quantization (auto-expanded, same storage as bin indices)
 * </ul>
 *
 * <p>SIMD-optimized scorers use the Java Vector API (JDK 25+) with platform-tuned patterns matching
 * Lucene's PanamaVectorUtilSupport for int8 dot products.
 *
 * @see org.apache.lucene.codecs.turboquant.TurboQuantFlatVectorsFormat
 */
package org.apache.lucene.codecs.turboquant;
