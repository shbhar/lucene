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
package org.apache.lucene.benchmark.jmh;

import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.util.VectorUtil;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Isolates {@link VectorUtil#expand8(int[])} and {@link VectorUtil#expand16(int[])} (the in-place
 * bit-spreading used by the lucene104 ForUtil postings codec) so the SIMD speedup is measured
 * directly rather than diluted by the rest of the block decode.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 4, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 3,
    jvmArgsAppend = {
      "-Xmx1g",
      "-Xms1g",
      "-XX:+AlwaysPreTouch",
      "--add-modules=jdk.incubator.vector"
    })
public class ExpandBenchmark {

  // ForUtil.BLOCK_SIZE
  private final int[] arr8 = new int[256];
  private final int[] arr16 = new int[256];

  @Setup(Level.Iteration)
  public void setup() {
    Random r = new Random(0);
    for (int i = 0; i < 256; ++i) {
      arr8[i] = r.nextInt();
      arr16[i] = r.nextInt();
    }
  }

  @Benchmark
  public void expand8(Blackhole bh) {
    VectorUtil.expand8(arr8);
    bh.consume(arr8);
  }

  @Benchmark
  public void expand16(Blackhole bh) {
    VectorUtil.expand16(arr16);
    bh.consume(arr16);
  }
}
