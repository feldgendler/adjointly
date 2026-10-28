/*
 * The arithmetic interface a differentiable function is written against, and the `plain`
 * implementation of it on ordinary numbers.
 */

import { primitives as P } from './primitives.ts';

/**
 * Scalar arithmetic over a value type `T`. Write a function once against this interface, generic
 * in `T`, and run it three ways: on plain numbers ({@link plain}), recording a tape for gradients
 * ({@link Tape}), or carrying tangents for directional derivatives ({@link Forward}).
 *
 * Every operation on a `T` must go through these methods. A `T` is opaque: for `Tape` and
 * `Forward` it is an index into storage, and `Math.exp(x)` or `x * y` on it computes nonsense
 * (the branded types make most such mistakes compile errors). Numbers enter through
 * {@link Arithmetic.constant} and leave through {@link Arithmetic.value}.
 *
 * The values every implementation computes come from the same {@link primitives} table, so a
 * function's value is bit-for-bit identical under all of them. Subgradient conventions (`abs` at
 * `0`, ties in `max`/`min`, `pow` with a non-positive base) are documented there.
 *
 * @typeParam T - the value type: `number` for `plain`, `Var` for `Tape`, `Dual` for `Forward`.
 *
 * @example
 * ```ts
 * import { type Arithmetic, plain, gradient } from 'adjointly';
 *
 * // f(x, y) = x * e^y + 3
 * function f<T>(A: Arithmetic<T>, [x, y]: readonly [T, T]): T {
 *   return A.addc(A.mul(x, A.exp(y)), 3);
 * }
 * f(plain, [2, 0]);   //=> 5
 * gradient(f, [2, 0]); //=> { value: 5, gradient: [1, 2] }
 * ```
 */
export interface Arithmetic<T> {
  /**
   * Lift a number into `T` as a constant: its derivative with respect to every input is zero.
   * On a `Tape` it still appends a node (constants are leaves of the tape).
   */
  constant(x: number): T;
  /**
   * The current numeric value of `a`. Use it to branch (`if (A.value(x) > 0)`), to log, or to
   * return a result. The number it returns carries no derivative information: anything computed
   * from it with ordinary arithmetic is a constant as far as differentiation is concerned.
   */
  value(a: T): number;

  /** `-a`. */
  neg(a: T): T;
  /** `a + b`. */
  add(a: T, b: T): T;
  /** `a - b`. */
  sub(a: T, b: T): T;
  /** `a * b`. */
  mul(a: T, b: T): T;
  /** `a / b`. */
  div(a: T, b: T): T;
  /** `1 / a`. */
  recip(a: T): T;
  /** `a * a`, one node instead of a `mul` of `a` with itself. */
  square(a: T): T;
  /** `sqrt(a)`; its derivative at `0` is `+Infinity`. */
  sqrt(a: T): T;
  /** `e^a`. */
  exp(a: T): T;
  /** `e^a - 1`, accurate for small `a`. */
  expm1(a: T): T;
  /** Natural logarithm. */
  log(a: T): T;
  /** `log(1 + a)`, accurate for small `a`. */
  log1p(a: T): T;
  /**
   * `a ** b`, differentiable in both arguments. The derivative with respect to `b` is `0` wherever
   * `a <= 0`; the derivative with respect to `a` is `0` wherever `b === 0`. For a fixed numeric
   * exponent use {@link Arithmetic.powc}; for `b = 2` prefer {@link Arithmetic.square}, for
   * `b = 0.5` prefer {@link Arithmetic.sqrt}.
   */
  pow(a: T, b: T): T;
  /** `|a|`; its derivative at `0` is `0`. */
  abs(a: T): T;
  /** The smaller of `a` and `b`; on a tie the derivative flows to `a`. */
  min(a: T, b: T): T;
  /** The larger of `a` and `b`; on a tie the derivative flows to `a`. */
  max(a: T, b: T): T;
  /** Sine. */
  sin(a: T): T;
  /** Cosine. */
  cos(a: T): T;
  /** Hyperbolic tangent. */
  tanh(a: T): T;
  /** Logistic sigmoid `1 / (1 + e^-a)`, stable for any `a`. */
  sigmoid(a: T): T;
  /** Softplus `log(1 + e^a)`, stable for any `a`; its derivative is `sigmoid(a)`. */
  softplus(a: T): T;

  /**
   * `a + c` for a plain number `c`: `add(a, constant(c))` without the constant's tape node. Same
   * value and derivative, bit for bit.
   */
  addc(a: T, c: number): T;
  /** `a - c` for a plain number `c`; `sub(a, constant(c))` in one node. */
  subc(a: T, c: number): T;
  /** `c - a` for a plain number `c`; `sub(constant(c), a)` in one node. */
  csub(c: number, a: T): T;
  /** `a * c` for a plain number `c`; `mul(a, constant(c))` in one node. */
  mulc(a: T, c: number): T;
  /** `a / c` for a plain number `c`; `div(a, constant(c))` in one node. */
  divc(a: T, c: number): T;
  /** `c / a` for a plain number `c`; `div(constant(c), a)` in one node. */
  cdiv(c: number, a: T): T;
  /**
   * `a ** c` for a plain number `c`; `pow(a, constant(c))` in one node, with the same conventions
   * (the derivative is `0` when `c === 0`).
   */
  powc(a: T, c: number): T;
}

/**
 * The operations with a plain-number operand, each the binary primitive `of` with one operand a
 * constant: the first one when `constantFirst`, otherwise the second. Their values and partials
 * are that primitive's, from the table.
 */
export const constantOperandOps = {
  addc: { of: 'add', constantFirst: false },
  subc: { of: 'sub', constantFirst: false },
  csub: { of: 'sub', constantFirst: true },
  mulc: { of: 'mul', constantFirst: false },
  divc: { of: 'div', constantFirst: false },
  cdiv: { of: 'div', constantFirst: true },
  powc: { of: 'pow', constantFirst: false },
} as const;

/** The names of the operations with a plain-number operand. */
export type ConstantOperandName = keyof typeof constantOperandOps;

/**
 * The inputs of a {@link ScalarFunction}: one `T` for each number in the point `X`. For a point
 * written as a literal, `[3, 4]`, that is the tuple `readonly [T, T]`, so destructured inputs need
 * no non-null assertions; for a `number[]` it is `readonly T[]`.
 */
export type Inputs<T, X extends readonly number[]> = { readonly [K in keyof X]: T };

/**
 * A function of several inputs written once against {@link Arithmetic}, the shape the
 * conveniences ({@link gradient}, {@link jvp}, {@link evaluate}, {@link checkGradient}) accept.
 * The input array has exactly as many elements as the point it is evaluated at; `X` is the type of
 * that point, so `ScalarFunction<[number, number]>` takes two inputs.
 *
 * @example
 * ```ts
 * const rosenbrock: ScalarFunction<[number, number]> = (A, [x, y]) =>
 *   A.add(
 *     A.square(A.csub(1, x)),
 *     A.mulc(A.square(A.sub(y, A.square(x))), 100),
 *   );
 * evaluate(rosenbrock, [1, 1]); //=> 0
 * gradient(rosenbrock, [0, 0]).gradient; //=> [-2, 0]
 * ```
 */
export type ScalarFunction<X extends readonly number[] = readonly number[]> = <T>(
  A: Arithmetic<T>,
  xs: Inputs<T, X>,
) => T;

/**
 * {@link Arithmetic} on ordinary numbers: computes values only, with nothing recorded. Use it to
 * evaluate a differentiable function at full speed when no derivative is needed.
 *
 * @example
 * ```ts
 * import { plain } from 'adjointly';
 * plain.add(plain.constant(1), plain.exp(0)); //=> 2
 * ```
 */
export const plain: Arithmetic<number> = {
  constant: (x) => x,
  value: (a) => a,
  neg: P.neg.value,
  add: P.add.value,
  sub: P.sub.value,
  mul: P.mul.value,
  div: P.div.value,
  recip: P.recip.value,
  square: P.square.value,
  sqrt: P.sqrt.value,
  exp: P.exp.value,
  expm1: P.expm1.value,
  log: P.log.value,
  log1p: P.log1p.value,
  pow: P.pow.value,
  abs: P.abs.value,
  min: P.min.value,
  max: P.max.value,
  sin: P.sin.value,
  cos: P.cos.value,
  tanh: P.tanh.value,
  sigmoid: P.sigmoid.value,
  softplus: P.softplus.value,
  // The constant-operand operations take their arguments in the same order as the primitive.
  addc: P.add.value,
  subc: P.sub.value,
  csub: P.sub.value,
  mulc: P.mul.value,
  divc: P.div.value,
  cdiv: P.div.value,
  powc: P.pow.value,
};
