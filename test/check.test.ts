/* finiteDifference and checkGradient. */

import { describe, expect, test } from 'vitest';
import {
  checkGradient,
  DEFAULT_STEP,
  evaluate,
  finiteDifference,
  logsumexp,
  type ScalarFunction,
} from '../src/index.ts';
import { expectAllClose } from './support/util.ts';

const smooth: ScalarFunction = (A, xs) =>
  A.add(
    logsumexp(
      A,
      xs.map((x) => A.mul(x, x)),
    ),
    A.sin(A.mul(xs[0]!, xs[1]!)),
  );

describe('finiteDifference', () => {
  test('approximates the gradient by central differences', () => {
    const g = finiteDifference((A, [x, y]) => A.mul(A.square(x), y), [3, -2]);
    expectAllClose(g, [-12, 9], 1e-9);
  });

  test('scales the step with the magnitude of each input', () => {
    // At x = 1e8 an absolute step of 6e-6 would be lost to rounding; a relative one is not.
    const g = finiteDifference((A, [x]) => A.square(x), [1e8]);
    expectAllClose(g, [2e8], 1e-6);
  });

  test('uses the given step: a huge step shows the truncation error of a cubic', () => {
    // central difference of x^3 with step h: 3x^2 + h^2
    expectAllClose(
      finiteDifference((A, [x]) => A.mul(x, A.square(x)), [0], 0.5),
      [0.25],
      1e-12,
    );
  });

  test('restores the point it perturbs', () => {
    const at = [1, 2];
    finiteDifference(smooth, at);
    expect(at).toEqual([1, 2]);
    expect(DEFAULT_STEP).toBeGreaterThan(0);
  });
});

describe('checkGradient', () => {
  test('passes a correct gradient', () => {
    const r = checkGradient(smooth, [0.3, -1.2, 2]);
    expect(r.ok).toBe(true);
    expect(r.analytic).toHaveLength(3);
    expect(r.numeric).toHaveLength(3);
    // Every component passes, and worst still names the closest call.
    expect(r.worst.index).toBeGreaterThanOrEqual(0);
    expect(r.worst.error).toBeLessThanOrEqual(r.worst.allowed);
    const ratios = r.analytic.map(
      (a, i) =>
        Math.abs(a - r.numeric[i]!) /
        (1e-8 + 1e-5 * Math.max(Math.abs(a), Math.abs(r.numeric[i]!))),
    );
    expect(r.worst.index).toBe(ratios.indexOf(Math.max(...ratios)));
    expect(r.value).toBe(evaluate(smooth, [0.3, -1.2, 2]));
  });

  test('catches a deliberately wrong derivative and names the worst component', () => {
    // Component 1 reads its own value back as a constant, so the tape misses half of
    // d(y^2)/dy: analytic y, numeric 2y.
    const wrong: ScalarFunction = (A, [x, y]) =>
      A.add(A.sin(x!), A.mul(y!, A.constant(A.value(y!))));
    const r = checkGradient(wrong, [0.5, 3]);
    expect(r.ok).toBe(false);
    expect(r.worst.index).toBe(1);
    expect(r.worst.analytic).toBe(3);
    expect(r.worst.numeric).toBeCloseTo(6, 6);
    expect(r.worst.error).toBeCloseTo(3, 6);
    expect(r.worst.allowed).toBe(1e-8 + 1e-5 * r.worst.numeric + r.worst.roundoff);
  });

  test('the worst component is the one furthest outside its allowance', () => {
    const wrong: ScalarFunction = (A, [x, y, z]) =>
      A.add(
        A.add(A.mul(x!, A.constant(A.value(x!))), A.square(y!)),
        A.mul(z!, A.constant(A.value(z!))),
      );
    // x: analytic 1, numeric 2 (error 1); z: analytic 10, numeric 20 (error 10, allowance 2e-4)
    const r = checkGradient(wrong, [1, 1, 10]);
    expect(r.worst.index).toBe(2);
  });

  test('worst is ranked by error relative to allowance, not by error', () => {
    // x: analytic 1, numeric 2: error 1 against an allowance of 2e-5, failing.
    // z: analytic 1e6 + 5, numeric 1e6 + 10: error 5 against an allowance of 10, passing.
    const f: ScalarFunction = (A, [x, z]) =>
      A.add(
        A.mul(x!, A.constant(A.value(x!))),
        A.add(A.mul(A.constant(1e6), z!), A.mul(A.constant(5), A.mul(z!, A.constant(A.value(z!))))),
      );
    const r = checkGradient(f, [1, 1]);
    expect(r.ok).toBe(false);
    expect(r.worst.index).toBe(0);
  });

  test('a component exactly at its allowance passes', () => {
    // f = x - value(x): analytic 1, but f is 0 everywhere, so the numeric derivative and its
    // rounding allowance are exactly 0 and the error is exactly 1.
    const f: ScalarFunction = (A, [x]) => A.sub(x!, A.constant(A.value(x!)));
    expect(checkGradient(f, [0.75]).numeric).toEqual([0]);
    expect(checkGradient(f, [0.75]).worst.roundoff).toBe(0);
    expect(checkGradient(f, [0.75], { rtol: 0, atol: 1 }).ok).toBe(true);
    expect(checkGradient(f, [0.75], { rtol: 0, atol: 0.999 }).ok).toBe(false);
  });

  test('an exact gradient passes when f is large, despite rounding in the differences', () => {
    const f: ScalarFunction = (A, [x, y]) =>
      A.add(A.add(A.constant(1e5), A.mul(A.constant(1e-3), x!)), A.sin(y!));
    const r = checkGradient(f, [0.7, 0.3]);
    expect(r.ok).toBe(true);
    expect(r.worst.roundoff).toBeGreaterThan(r.worst.error);
    // ...and a wrong one still fails there.
    const wrong: ScalarFunction = (A, [x]) =>
      A.add(A.constant(1e5), A.mul(x!, A.constant(A.value(x!))));
    expect(checkGradient(wrong, [1]).ok).toBe(false);
  });

  test('the rounding allowance is about eps |f| / step', () => {
    const f: ScalarFunction = (A, [x]) => A.add(A.constant(1e5), x!);
    const r = checkGradient(f, [0]);
    const step = 2 * DEFAULT_STEP;
    expect(r.worst.roundoff).toBeCloseTo((4 * Number.EPSILON * 2e5) / step, 12);
    expect(r.worst.allowed).toBe(1e-8 + 1e-5 * Math.max(1, r.numeric[0]!) + r.worst.roundoff);
  });

  test('tolerances are honoured', () => {
    const wrong: ScalarFunction = (A, [x]) => A.mul(x!, A.constant(1.001 * A.value(x!)));
    // analytic 1.001x vs numeric 2.002x at x=1: relative error ~0.5
    expect(checkGradient(wrong, [1]).ok).toBe(false);
    expect(checkGradient(wrong, [1], { rtol: 0.6 }).ok).toBe(true);
    expect(checkGradient(wrong, [1], { atol: 2 }).ok).toBe(true);
    expect(checkGradient(wrong, [1], { rtol: 0.4 }).ok).toBe(false);
  });

  test('the step is honoured', () => {
    const cube: ScalarFunction = (A, [x]) => A.mul(x!, A.square(x!));
    expect(checkGradient(cube, [0], { h: 0.5 }).numeric).toEqual([0.25]);
  });

  test('a NaN anywhere fails the check and ranks worst', () => {
    const r = checkGradient((A, [x, y]) => A.add(A.square(x), A.log(y)), [2, -1]);
    expect(r.ok).toBe(false);
    // f is NaN everywhere near this point, so even x's numeric derivative is NaN.
    expect(r.numeric.every(Number.isNaN)).toBe(true);
    expect(r.worst.index).toBe(0);
    expect(r.worst.analytic).toBe(4);
    const one = checkGradient((A, [y]) => A.log(y), [-1]);
    expect(one.ok).toBe(false);
    expect(one.worst.index).toBe(0);
    expect(one.worst.analytic).toBe(-1);
    expect(one.worst.numeric).toBeNaN();
  });

  test('with zero tolerances and exact zero components, worst still names a component', () => {
    const r = checkGradient((A) => A.constant(0), [2, 3], { rtol: 0, atol: 0 });
    expect(r.ok).toBe(true);
    expect(r.worst.index).toBe(0);
    expect(r.worst.error).toBe(0);
  });

  test('a function of no inputs passes trivially', () => {
    const r = checkGradient((A) => A.constant(1), []);
    expect(r).toEqual({
      ok: true,
      value: 1,
      analytic: [],
      numeric: [],
      worst: { index: -1, analytic: NaN, numeric: NaN, error: 0, allowed: 0, roundoff: 0 },
    });
  });
});
