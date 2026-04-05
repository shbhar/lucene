/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.
 */
package org.apache.lucene.codecs.turboquant;

import org.apache.lucene.tests.util.LuceneTestCase;

/** Tests for the Fast Walsh-Hadamard Transform. */
public class TestFWHT extends LuceneTestCase {

  public void testTransformPreservesNorm() {
    // Normalized FWHT (H/sqrt(n)) is orthogonal, so it preserves L2 norm
    float[] v = randomUnitVector(256);
    float normBefore = norm(v);
    FWHT.transform(v);
    float normAfter = norm(v);
    assertEquals(normBefore, normAfter, 1e-4f);
  }

  public void testTransformIsInvolution() {
    // Normalized WHT applied twice = identity (H/sqrt(n) * H/sqrt(n) = I)
    float[] v = randomUnitVector(512);
    float[] original = v.clone();
    FWHT.transform(v);
    FWHT.transform(v);
    for (int i = 0; i < v.length; i++) {
      assertEquals(original[i], v[i], 1e-4f);
    }
  }

  public void testSignFlipsAreReversible() {
    float[] v = randomUnitVector(128);
    float[] original = v.clone();
    float[] signs = new float[128];
    for (int i = 0; i < 128; i++) signs[i] = random().nextBoolean() ? 1f : -1f;
    FWHT.applySignFlips(v, signs);
    FWHT.applySignFlips(v, signs); // applying same signs twice = identity
    for (int i = 0; i < v.length; i++) {
      assertEquals(original[i], v[i], 1e-6f);
    }
  }

  public void testTransformPreservesDotProduct() {
    // Orthogonal transform preserves inner products
    int dim = 256;
    float[] a = randomUnitVector(dim);
    float[] b = randomUnitVector(dim);
    float dotBefore = dot(a, b);
    float[] signs = new float[dim];
    for (int i = 0; i < dim; i++) signs[i] = random().nextBoolean() ? 1f : -1f;
    FWHT.applySignFlips(a, signs);
    FWHT.applySignFlips(b, signs);
    FWHT.transform(a);
    FWHT.transform(b);
    float dotAfter = dot(a, b);
    assertEquals(dotBefore, dotAfter, 1e-4f);
  }

  private float[] randomUnitVector(int dim) {
    float[] v = new float[dim];
    float n = 0;
    for (int i = 0; i < dim; i++) { v[i] = (float) random().nextGaussian(); n += v[i] * v[i]; }
    n = (float) Math.sqrt(n);
    for (int i = 0; i < dim; i++) v[i] /= n;
    return v;
  }

  private float norm(float[] v) {
    float s = 0; for (float x : v) s += x * x; return (float) Math.sqrt(s);
  }

  private float dot(float[] a, float[] b) {
    float s = 0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
  }
}
