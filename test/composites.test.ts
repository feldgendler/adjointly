/* sum, dot, logsumexp and softmax: values, gradients and numerical stability. */

import { describe, expect, test } from 'vitest';
import {
  dot,
  Forward,
  gradient,
  logsumexp,
  plain,
  softmax,
  sum,
  Tape,
  type Var,
} from '../src/index.ts';
import { expectAllClose, expectClose } from './support/util.ts';

describe('sum and dot', () => {
  test('values', () => {
    expect(sum(plain, [])).toBe(0);
    expect(sum(plain, [4])).toBe(4);
    expect(sum(plain, [1, 2, 3])).toBe(6);
    expect(dot(plain, [], [])).toBe(0);
    expect(dot(plain, [2], [3])).toBe(6);
    expect(dot(plain, [1, 2, 3], [4, 5, 6])).toBe(32);
  });

  test('gradients', () => {
    expect(gradient((A, xs) => sum(A, xs), [1, 2, 3]).gradient).toEqual([1, 1, 1]);
    const g = gradient((A, xs) => dot(A, xs.slice(0, 2), xs.slice(2)), [1, 2, 3, 4]).gradient;
    expect(g).toEqual([3, 4, 1, 2]);
  });

  test('dot rejects lists of different lengths', () => {
    expect(() => dot(plain, [1], [1, 2])).toThrow(/lengths differ \(1 and 2\)/);
  });

  test('an empty sum on a tape is a constant node', () => {
    const t = new Tape();
    const s = sum(t, []);
    expect(t.size).toBe(1);
    expect(t.value(s)).toBe(0);
  });
});

describe('logsumexp', () => {
  const naive = (xs: number[]): number => Math.log(xs.reduce((s, x) => s + Math.exp(x), 0));

  test('matches the naive formula where that does not overflow', () => {
    expectClose(logsumexp(plain, [1, 2, 3]), naive([1, 2, 3]), 1e-15);
    expectClose(logsumexp(plain, [-1, 0.5]), naive([-1, 0.5]), 1e-15);
    expect(logsumexp(plain, [7])).toBe(7);
  });

  test('is stable for large and very negative inputs', () => {
    expectClose(logsumexp(plain, [1000, 1000]), 1000 + Math.LN2, 1e-15);
    expectClose(logsumexp(plain, [-1000, -1000]), -1000 + Math.LN2, 1e-15);
    expect(logsumexp(plain, [1000, -Infinity])).toBe(1000);
  });

  test('infinite inputs: the limits of the finite case', () => {
    const lse = (at: number[]) => gradient((A, xs) => logsumexp(A, xs), at);
    // All -Infinity: as the inputs fall together, the weights stay equal.
    expect(lse([-Infinity, -Infinity])).toEqual({ value: -Infinity, gradient: [0.5, 0.5] });
    // A +Infinity input takes all the weight; several share it.
    expect(lse([Infinity, 0, -1])).toEqual({ value: Infinity, gradient: [1, 0, 0] });
    expect(lse([Infinity, 0, Infinity])).toEqual({ value: Infinity, gradient: [0.5, 0, 0.5] });
    // A -Infinity input among finite ones has weight 0.
    expect(lse([-Infinity, 0])).toEqual({ value: 0, gradient: [0, 1] });
    // NaN anywhere is NaN.
    expect(lse([NaN, 0]).value).toBeNaN();
    expect(lse([NaN, Infinity]).value).toBeNaN();
  });

  test('its gradient is the softmax, exactly through the shift', () => {
    const at = [1000, 1001, 999];
    const g = gradient((A, xs) => logsumexp(A, xs), at).gradient;
    expectAllClose(g, softmax(plain, at), 1e-15);
    expectClose(
      g.reduce((s, v) => s + v, 0),
      1,
      1e-15,
    );
  });

  test('rejects an empty list', () => {
    expect(() => logsumexp(plain, [])).toThrow(/empty/);
  });
});

describe('softmax', () => {
  test('values sum to 1 and are stable', () => {
    expectAllClose(softmax(plain, [0, Math.log(3)]), [0.25, 0.75], 1e-15);
    expectAllClose(softmax(plain, [1000, 1000]), [0.5, 0.5]);
    expectAllClose(softmax(plain, [-1000, -1000 + Math.log(3)]), [0.25, 0.75], 1e-12);
    expect(softmax(plain, [5])).toEqual([1]);
  });

  test('infinite inputs: the limits of the finite case, with a zero Jacobian', () => {
    expect(softmax(plain, [-Infinity, -Infinity])).toEqual([0.5, 0.5]);
    expect(softmax(plain, [Infinity, 0, -Infinity])).toEqual([1, 0, 0]);
    expect(softmax(plain, [Infinity, 0, Infinity])).toEqual([0.5, 0, 0.5]);
    expect(softmax(plain, [-Infinity, 0])).toEqual([0, 1]);
    expect(softmax(plain, [NaN, Infinity]).every(Number.isNaN)).toBe(true);
    const t = new Tape();
    const xs = t.inputs([Infinity, 0]);
    t.vjp(softmax(t, xs), [1, 2]);
    expect(Array.from(t.adjoints(xs))).toEqual([0, 0]);
  });

  test('Jacobian is diag(p) - p p^T, by reverse and forward mode', () => {
    const at = [0.3, -1.2, 2.0, 0.7];
    const p = softmax(plain, at);
    for (let i = 0; i < at.length; i++) {
      const t = new Tape();
      const xs = t.inputs(at);
      const ys = softmax(t, xs);
      t.backward(ys[i]!);
      const row = t.adjoints(xs);
      for (let j = 0; j < at.length; j++) {
        const want = (i === j ? p[i]! : 0) - p[i]! * p[j]!;
        expectClose(row[j]!, want, 1e-12, 1e-16);
        const F = new Forward();
        const fs = softmax(
          F,
          at.map((x, k) => F.input(x, k === j ? 1 : 0)),
        );
        expectClose(F.tangent(fs[i]!), want, 1e-12, 1e-16);
      }
    }
  });

  test('rejects an empty list', () => {
    expect(() => softmax(plain, [])).toThrow(/empty/);
  });

  test('the shift is a constant on the tape', () => {
    const t = new Tape();
    const xs: Var[] = t.inputs([1, 2]);
    softmax(t, xs);
    // 2 inputs, 1 shift constant, 2 (sub, exp) pairs, 1 add, 1 recip, 2 mul
    expect(t.size).toBe(2 + 1 + 4 + 1 + 1 + 2);
  });
});
