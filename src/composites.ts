/*
 * Composite operations built from primitives, generic over any {@link Arithmetic}: they record
 * ordinary primitive nodes on a tape and need no derivative rules of their own.
 */

import type { Arithmetic } from './arithmetic.ts';

/**
 * The sum of `xs`, added left to right; `A.constant(0)` for an empty list.
 *
 * @example
 * ```ts
 * sum(plain, [1, 2, 3]); //=> 6
 * ```
 */
export function sum<T>(A: Arithmetic<T>, xs: readonly T[]): T {
  if (xs.length === 0) return A.constant(0);
  let s: T = xs[0]!;
  for (let i = 1; i < xs.length; i++) s = A.add(s, xs[i]!);
  return s;
}

/**
 * The dot product `sum_i xs[i] * ys[i]`; `A.constant(0)` for empty lists. Throws if the lengths
 * differ.
 *
 * @example
 * ```ts
 * dot(plain, [1, 2], [3, 4]); //=> 11
 * ```
 */
export function dot<T>(A: Arithmetic<T>, xs: readonly T[], ys: readonly T[]): T {
  if (xs.length !== ys.length) {
    throw new RangeError(`dot: lengths differ (${xs.length} and ${ys.length})`);
  }
  if (xs.length === 0) return A.constant(0);
  let s = A.mul(xs[0]!, ys[0]!);
  for (let i = 1; i < xs.length; i++) s = A.add(s, A.mul(xs[i]!, ys[i]!));
  return s;
}

/**
 * The largest of the values of `xs`, as a plain number, for use as a constant shift. `0` when that
 * would be infinite or `NaN`, so that shifting by it never produces `Infinity - Infinity`.
 */
function shiftFor<T>(A: Arithmetic<T>, xs: readonly T[]): number {
  let m = -Infinity;
  for (const x of xs) m = Math.max(m, A.value(x));
  return Number.isFinite(m) ? m : 0;
}

/**
 * `log(sum_i exp(xs[i]))`, computed stably as `m + log(sum_i exp(xs[i] - m))` where `m` is the
 * largest value. `m` enters as a constant: the identity holds for any `m`, so its derivative
 * through `m` is zero and leaving it out keeps the gradient exact (it is `softmax(xs)`).
 * Throws on an empty list.
 *
 * @example
 * ```ts
 * // The naive log(exp(1000) + exp(1000)) overflows to Infinity.
 * logsumexp(plain, [1000, 1000]); //=> 1000 + log(2)
 * ```
 */
export function logsumexp<T>(A: Arithmetic<T>, xs: readonly T[]): T {
  if (xs.length === 0) throw new RangeError('logsumexp: empty list');
  const m = shiftFor(A, xs);
  const c = A.constant(m);
  let s: T = A.exp(A.sub(xs[0]!, c));
  for (let i = 1; i < xs.length; i++) s = A.add(s, A.exp(A.sub(xs[i]!, c)));
  return A.add(c, A.log(s));
}

/**
 * The softmax `exp(xs[i]) / sum_j exp(xs[j])`, computed stably by first subtracting the largest
 * value as a constant (softmax is invariant to the shift, so the gradient stays exact). Throws on
 * an empty list.
 *
 * @example
 * ```ts
 * softmax(plain, [0, Math.log(3)]); //=> [0.25, 0.75]
 * ```
 */
export function softmax<T>(A: Arithmetic<T>, xs: readonly T[]): T[] {
  if (xs.length === 0) throw new RangeError('softmax: empty list');
  const c = A.constant(shiftFor(A, xs));
  const es = xs.map((x) => A.exp(A.sub(x, c)));
  const inv = A.recip(sum(A, es));
  return es.map((e) => A.mul(e, inv));
}
