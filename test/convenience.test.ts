/* evaluate, gradient and jvp; and control flow inside differentiated functions. */

import { describe, expect, test } from 'vitest';
import {
  type Arithmetic,
  evaluate,
  Forward,
  gradient,
  jvp,
  type ScalarFunction,
  Tape,
} from '../src/index.ts';
import { expectClose } from './support/util.ts';

const f: ScalarFunction = (A, [x, y]) => A.mul(A.sin(x!), A.exp(y!));

describe('evaluate, gradient, jvp', () => {
  test('agree on the value and on the derivative', () => {
    const at = [0.7, -0.4];
    const v = evaluate(f, at);
    expect(v).toBe(Math.sin(0.7) * Math.exp(-0.4));
    const g = gradient(f, at);
    expect(g.value).toBe(v);
    expectClose(g.gradient[0]!, Math.cos(0.7) * Math.exp(-0.4));
    expectClose(g.gradient[1]!, v);
    const d = jvp(f, at, [2, -3]);
    expect(d.value).toBe(v);
    expectClose(d.derivative, 2 * g.gradient[0]! - 3 * g.gradient[1]!);
  });

  test('gradient reuses a given tape, clearing it first', () => {
    const t = new Tape({ capacity: 2 });
    t.input(99);
    const r1 = gradient(f, [1, 2], t);
    const size = t.size;
    const r2 = gradient(f, [1, 2], t);
    expect(t.size).toBe(size);
    expect(r2).toEqual(r1);
  });

  test('jvp reuses a given Forward, clearing it first', () => {
    const F = new Forward();
    F.input(5, 5);
    expect(jvp(f, [1, 2], [1, 0], F)).toEqual(jvp(f, [1, 2], [1, 0]));
    expect(F.size).toBe(5);
  });

  test('jvp rejects a direction of the wrong length', () => {
    expect(() => jvp(f, [1, 2], [1])).toThrow(/2 inputs but a direction of length 1/);
  });

  test('a function of no inputs, and one returning an input unchanged', () => {
    expect(gradient((A) => A.constant(3), [])).toEqual({ value: 3, gradient: [] });
    expect(gradient((_A, [x]) => x!, [4])).toEqual({ value: 4, gradient: [1] });
    expect(jvp((_A, [x]) => x!, [4], [2])).toEqual({ value: 4, derivative: 2 });
  });
});

describe('control flow', () => {
  // A piecewise function: x^2 for x < 1, 2x - 1 for x >= 1 (continuous, C1 at 1).
  const piecewise: ScalarFunction = (A, [x]) =>
    A.value(x!) < 1 ? A.square(x!) : A.sub(A.mul(A.constant(2), x!), A.constant(1));

  test('branches on A.value: the tape records the branch that ran', () => {
    expect(gradient(piecewise, [0.5]).gradient).toEqual([1]);
    expect(gradient(piecewise, [3]).gradient).toEqual([2]);
  });

  test('at a branch point the gradient is the taken branch’s one-sided derivative', () => {
    // |x| written by hand: the branch for x >= 0 runs at 0, so the derivative there is +1,
    // unlike the abs primitive's 0.
    const hand: ScalarFunction = (A, [x]) => (A.value(x!) >= 0 ? x! : A.neg(x!));
    expect(gradient(hand, [0]).gradient).toEqual([1]);
    expect(gradient((A, [x]) => A.abs(x!), [0]).gradient).toEqual([0]);
  });

  test('a loop whose trip count depends on values: Newton’s method for sqrt(a)', () => {
    // Differentiating through the iterations gives d sqrt(a)/da = 1 / (2 sqrt(a)).
    function newtonSqrt<T>(A: Arithmetic<T>, [a]: readonly T[]): T {
      let x = A.constant(1);
      for (let i = 0; i < 100; i++) {
        const next = A.mul(A.constant(0.5), A.add(x, A.div(a!, x)));
        if (Math.abs(A.value(next) - A.value(x)) < 1e-15) return next; // early return
        x = next;
      }
      return x;
    }
    const r = gradient(newtonSqrt, [2]);
    expectClose(r.value, Math.SQRT2, 1e-15);
    expectClose(r.gradient[0]!, 1 / (2 * Math.SQRT2), 1e-12);
    expectClose(jvp(newtonSqrt, [2], [1]).derivative, 1 / (2 * Math.SQRT2), 1e-12);
  });

  test('a number read with A.value and used again is a constant: its derivative is lost', () => {
    // The classic mistake: x * value(x) has gradient x, not 2x.
    const leaky: ScalarFunction = (A, [x]) => A.mul(x!, A.constant(A.value(x!)));
    expect(gradient(leaky, [3]).gradient).toEqual([3]);
  });
});
