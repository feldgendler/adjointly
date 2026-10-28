/*
 * Numerical checking: central finite differences, and a gradient check comparing them with
 * reverse mode.
 */

import { plain, type ScalarFunction } from './arithmetic.ts';
import { gradient } from './convenience.ts';

/**
 * Default relative step for {@link finiteDifference}: about the cube root of machine epsilon,
 * which balances the central difference's `O(h^2)` truncation error against rounding error.
 */
export const DEFAULT_STEP = 6e-6;

/**
 * The gradient of `f` at `at` by central differences on plain numbers:
 * `(f(x + h_i e_i) - f(x - h_i e_i)) / (2 h_i)` with `h_i = h * max(1, |at[i]|)`.
 * Costs two evaluations per input; expect agreement with the true gradient to roughly 1e-8
 * relative for a smooth, well-scaled function, and nothing useful across a kink or a branch.
 *
 * @param h - relative step, default {@link DEFAULT_STEP}.
 *
 * @example
 * ```ts
 * finiteDifference((A, [x]) => A.square(x!), [3]); //=> ~[6]
 * ```
 */
export function finiteDifference(
  f: ScalarFunction,
  at: readonly number[],
  h: number = DEFAULT_STEP,
): number[] {
  const x = at.slice(); // perturbed one component at a time
  return at.map((xi, i) => {
    const step = h * Math.max(1, Math.abs(xi));
    x[i] = xi + step;
    const up = f(plain, x);
    x[i] = xi - step;
    const down = f(plain, x);
    x[i] = xi;
    // Divide by the step actually taken, which rounding may have changed.
    return (up - down) / (xi + step - (xi - step));
  });
}

/** Options for {@link checkGradient}. */
export interface CheckGradientOptions {
  /** Relative finite-difference step; default {@link DEFAULT_STEP}. */
  readonly h?: number;
  /** Relative tolerance; default `1e-5`. */
  readonly rtol?: number;
  /** Absolute tolerance; default `1e-8`. */
  readonly atol?: number;
}

/** The result of {@link checkGradient}. */
export interface GradientCheck {
  /** Whether every component is within tolerance (and none is `NaN`). */
  readonly ok: boolean;
  /** `f(at)`. */
  readonly value: number;
  /** The gradient by reverse mode. */
  readonly analytic: number[];
  /** The gradient by central differences. */
  readonly numeric: number[];
  /**
   * The component furthest out of tolerance relative to its allowance (`-1` index when `at` is
   * empty). `error` is `|analytic - numeric|`; `allowed` is `atol + rtol * max(|analytic|, |numeric|)`.
   */
  readonly worst: {
    readonly index: number;
    readonly analytic: number;
    readonly numeric: number;
    readonly error: number;
    readonly allowed: number;
  };
}

/**
 * Compare `f`'s reverse-mode gradient at `at` with central finite differences. Component `i`
 * passes when `|analytic - numeric| <= atol + rtol * max(|analytic|, |numeric|)`.
 *
 * Use it in your own tests to verify a function written against `Arithmetic`, typically at a few
 * random points away from kinks and branch boundaries, where finite differences are meaningless.
 *
 * @example
 * ```ts
 * const loss: ScalarFunction = (A, xs) => logsumexp(A, xs.map((x) => A.square(x)));
 * const r = checkGradient(loss, [0.3, -1.2, 2]);
 * if (!r.ok) {
 *   const { index, analytic, numeric } = r.worst;
 *   throw new Error(`component ${String(index)}: ${String(analytic)} vs ${String(numeric)}`);
 * }
 * r.ok; //=> true
 * ```
 */
export function checkGradient(
  f: ScalarFunction,
  at: readonly number[],
  options: CheckGradientOptions = {},
): GradientCheck {
  const { h = DEFAULT_STEP, rtol = 1e-5, atol = 1e-8 } = options;
  const { value, gradient: analytic } = gradient(f, at);
  const numeric = finiteDifference(f, at, h);
  let ok = true;
  let worst = { index: -1, analytic: NaN, numeric: NaN, error: 0, allowed: 0 };
  let worstRatio = -1;
  for (let i = 0; i < analytic.length; i++) {
    const a = analytic[i]!,
      n = numeric[i]!;
    const error = Math.abs(a - n);
    const allowed = atol + rtol * Math.max(Math.abs(a), Math.abs(n));
    // NaN compares false, so a NaN error ranks as infinitely bad and fails.
    const ratio = Number.isNaN(error) ? Infinity : error / allowed;
    if (!(error <= allowed)) ok = false;
    if (ratio > worstRatio) {
      worstRatio = ratio;
      worst = { index: i, analytic: a, numeric: n, error, allowed };
    }
  }
  return { ok, value, analytic, numeric, worst };
}
