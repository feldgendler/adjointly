/*
 * One-call entry points for a {@link ScalarFunction}: value, gradient, directional derivative.
 */

import { plain, type ScalarFunction } from './arithmetic.ts';
import { Forward } from './forward.ts';
import { Tape } from './tape.ts';

/** The result of {@link gradient}. */
export interface GradientResult {
  /** `f(at)`. */
  readonly value: number;
  /** `df/d(at[i])` for each `i`. */
  readonly gradient: number[];
}

/** The result of {@link jvp}. */
export interface JvpResult {
  /** `f(at)`. */
  readonly value: number;
  /** The directional derivative `sum_i df/d(at[i]) * direction[i]`. */
  readonly derivative: number;
}

/**
 * Evaluate `f` at `at` on plain numbers, recording nothing. The value is bit-for-bit the one
 * {@link gradient} and {@link jvp} report.
 *
 * @example
 * ```ts
 * evaluate((A, [x]) => A.exp(x!), [0]); //=> 1
 * ```
 */
export function evaluate(f: ScalarFunction, at: readonly number[]): number {
  return f(plain, at);
}

/**
 * The value and gradient of `f` at `at`, by reverse mode: one forward pass recording a tape and
 * one backward sweep, whatever the number of inputs.
 *
 * Pass `tape` to reuse its storage across calls in a loop; it is cleared first and left holding
 * this evaluation.
 *
 * @example
 * ```ts
 * gradient((A, [x, y]) => A.mul(x!, y!), [3, 4]); //=> { value: 12, gradient: [4, 3] }
 * ```
 */
export function gradient(
  f: ScalarFunction,
  at: readonly number[],
  tape: Tape = new Tape(),
): GradientResult {
  tape.clear();
  const xs = tape.inputs(at);
  const y = f(tape, xs);
  tape.backward(y);
  return { value: tape.value(y), gradient: xs.map((x) => tape.adjoint(x)) };
}

/**
 * The value of `f` at `at` and its derivative along `direction` (a Jacobian-vector product), by
 * forward mode: one pass, no tape. Throws if `direction` and `at` differ in length.
 *
 * @example
 * ```ts
 * jvp((A, [x, y]) => A.mul(x!, y!), [3, 4], [1, 0]); //=> { value: 12, derivative: 4 }
 * ```
 */
export function jvp(
  f: ScalarFunction,
  at: readonly number[],
  direction: readonly number[],
  forward: Forward = new Forward(),
): JvpResult {
  if (direction.length !== at.length) {
    throw new RangeError(`jvp: ${at.length} inputs but a direction of length ${direction.length}`);
  }
  forward.clear();
  const xs = at.map((x, i) => forward.input(x, direction[i]!));
  const y = f(forward, xs);
  return { value: forward.value(y), derivative: forward.tangent(y) };
}
