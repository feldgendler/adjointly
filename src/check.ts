/*
 * Numerical checking: central finite differences, and a gradient check comparing them with
 * reverse mode.
 */

import { type Inputs, plain, type ScalarFunction } from './arithmetic.ts';
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
 * finiteDifference((A, [x]) => A.square(x), [3]); //=> ~[6]
 * ```
 */
export function finiteDifference<const X extends readonly number[]>(
  f: ScalarFunction<X>,
  at: X,
  h: number = DEFAULT_STEP,
): number[] {
  return centralDifferences(f, at, h).map((d) => d.slope);
}

/** Units in the last place of error allowed for in each evaluation of `f`. */
const ROUNDING_ULPS = 4;

/**
 * Each component's central difference, and an estimate of the error that rounding in the two
 * evaluations of `f` puts into it: `ROUNDING_ULPS * eps * (|f(x + h)| + |f(x - h)|) / (2 h)`.
 */
function centralDifferences<X extends readonly number[]>(
  f: ScalarFunction<X>,
  at: X,
  h: number,
): { slope: number; roundoff: number }[] {
  const x: number[] = at.slice(); // perturbed one component at a time
  const point = x as unknown as Inputs<number, X>; // the same array, seen as f's inputs
  return at.map((xi, i) => {
    const step = h * Math.max(1, Math.abs(xi));
    x[i] = xi + step;
    const up = f(plain, point);
    x[i] = xi - step;
    const down = f(plain, point);
    x[i] = xi;
    // Divide by the step actually taken, which rounding may have changed.
    const span = xi + step - (xi - step);
    return {
      slope: (up - down) / span,
      roundoff: (ROUNDING_ULPS * Number.EPSILON * (Math.abs(up) + Math.abs(down))) / span,
    };
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
   * empty). `error` is `|analytic - numeric|`; `roundoff` the estimated rounding error of the
   * finite difference; `allowed` is `atol + rtol * max(|analytic|, |numeric|) + roundoff`.
   */
  readonly worst: {
    readonly index: number;
    readonly analytic: number;
    readonly numeric: number;
    readonly error: number;
    readonly allowed: number;
    readonly roundoff: number;
  };
}

/**
 * Compare `f`'s reverse-mode gradient at `at` with central finite differences. Component `i`
 * passes when `|analytic - numeric| <= atol + rtol * max(|analytic|, |numeric|) + roundoff`.
 *
 * `roundoff` allows for the rounding error of the finite difference itself, which grows with
 * `|f|` and shrinks with the step: `4 * eps * (|f(x + h)| + |f(x - h)|) / (2 h)`, four units in the
 * last place for each evaluation. Without it an exact gradient of a function whose value is large
 * next to its slope (`1e5 + 1e-3 * x`) would fail.
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
export function checkGradient<const X extends readonly number[]>(
  f: ScalarFunction<X>,
  at: X,
  options: CheckGradientOptions = {},
): GradientCheck {
  const { h = DEFAULT_STEP, rtol = 1e-5, atol = 1e-8 } = options;
  const { value, gradient: analytic } = gradient(f, at);
  const differences = centralDifferences(f, at, h);
  const numeric = differences.map((d) => d.slope);
  let ok = true;
  let worst = { index: -1, analytic: NaN, numeric: NaN, error: 0, allowed: 0, roundoff: 0 };
  let worstRatio = -1;
  for (let i = 0; i < analytic.length; i++) {
    const a = analytic[i]!;
    const n = numeric[i]!;
    const roundoff = differences[i]!.roundoff;
    const error = Math.abs(a - n);
    const allowed = atol + rtol * Math.max(Math.abs(a), Math.abs(n)) + roundoff;
    // A NaN error ranks as infinitely bad (and fails, NaN comparing false); an exact match ranks 0
    // even against a zero allowance, where 0 / 0 would be NaN.
    const ratio = Number.isNaN(error) ? Infinity : error === 0 ? 0 : error / allowed;
    if (!(error <= allowed)) ok = false;
    if (ratio > worstRatio) {
      worstRatio = ratio;
      worst = { index: i, analytic: a, numeric: n, error, allowed, roundoff };
    }
  }
  return { ok, value, analytic, numeric, worst };
}
