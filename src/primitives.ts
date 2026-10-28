/*
 * The table of primitive operations: the single source of truth for every operation's value and
 * its local partial derivatives.
 *
 * `plain`, {@link Tape} and {@link Forward} all compute a primitive's value by calling the same
 * `value` function from this table, so a program evaluates to bit-for-bit the same number under
 * all three. They differ only in what they do with the partials: `plain` ignores them, `Tape`
 * records them for the backward sweep, `Forward` multiplies them into tangents on the spot.
 *
 * Each partial receives the operation's own result `y` as well as its arguments, so derivatives
 * that are cheapest in terms of the result (`exp`, `tanh`, `sqrt`, ...) reuse it instead of
 * recomputing.
 */

/** A primitive of one argument: `y = value(x)`, `dy/dx = deriv(x, y)`. */
export interface UnaryPrimitive {
  /** The primitive's name, equal to its key in {@link primitives} and its method on `Arithmetic`. */
  readonly name: string;
  /** The value of the operation at `x`. */
  readonly value: (x: number) => number;
  /** The derivative `dy/dx` at `x`, given `y = value(x)`. */
  readonly deriv: (x: number, y: number) => number;
}

/** A primitive of two arguments: `y = value(a, b)`, with partials `da = dy/da` and `db = dy/db`. */
export interface BinaryPrimitive {
  /** The primitive's name, equal to its key in {@link primitives} and its method on `Arithmetic`. */
  readonly name: string;
  /** The value of the operation at `(a, b)`. */
  readonly value: (a: number, b: number) => number;
  /** The partial derivative `dy/da` at `(a, b)`, given `y = value(a, b)`. */
  readonly da: (a: number, b: number, y: number) => number;
  /** The partial derivative `dy/db` at `(a, b)`, given `y = value(a, b)`. */
  readonly db: (a: number, b: number, y: number) => number;
}

function unary(
  name: string,
  value: (x: number) => number,
  deriv: (x: number, y: number) => number,
): UnaryPrimitive {
  return Object.freeze({ name, value, deriv });
}

function binary(
  name: string,
  value: (a: number, b: number) => number,
  da: (a: number, b: number, y: number) => number,
  db: (a: number, b: number, y: number) => number,
): BinaryPrimitive {
  return Object.freeze({ name, value, da, db });
}

/**
 * Logistic sigmoid `1 / (1 + e^-x)`, evaluated without overflow for any `x`: `exp` is only ever
 * called on `-|x|`.
 */
function sigmoidValue(x: number): number {
  // Stryker disable next-line EqualityOperator: at x = 0 both branches give exactly 0.5.
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * `sigmoid'(x) = sigmoid(x) * sigmoid(-x) = e / (1 + e)^2` with `e = exp(-|x|)`. Computed from `x`
 * rather than as `y * (1 - y)`, which loses all precision for large positive `x` where `y` rounds
 * to 1.
 */
function sigmoidDeriv(x: number): number {
  const e = Math.exp(-Math.abs(x));
  const s = 1 + e;
  return e / (s * s);
}

/**
 * Every primitive, keyed by name. Subgradient conventions at points where the derivative does not
 * exist are part of the contract:
 *
 * | primitive | convention |
 * |---|---|
 * | `abs` | derivative at `0` is `0` |
 * | `max`, `min` | on a tie (`a === b`) the whole derivative goes to the first argument `a` |
 * | `pow` | `d/db` is `0` wherever `a <= 0`; `d/da` is `0` wherever `b === 0` |
 * | `sqrt` | derivative at `0` is `+Infinity` (the true one-sided limit) |
 *
 * Everywhere else the partials are the ordinary analytic derivatives, and outside a function's
 * domain (`log` of a negative, `sqrt` of a negative) the value and partials are whatever IEEE
 * arithmetic produces, usually `NaN`.
 *
 * @example
 * ```ts
 * import { primitives } from 'adjointly';
 * const x = 0.5;
 * const y = primitives.tanh.value(x); //=> tanh(0.5)
 * const dy = primitives.tanh.deriv(x, y); //=> 1 - tanh(0.5) ** 2
 * ```
 */
export const primitives = Object.freeze({
  /** `-x`. */
  neg: unary(
    'neg',
    (x) => -x,
    () => -1,
  ),
  /** `a + b`. */
  add: binary(
    'add',
    (a, b) => a + b,
    () => 1,
    () => 1,
  ),
  /** `a - b`. */
  sub: binary(
    'sub',
    (a, b) => a - b,
    () => 1,
    () => -1,
  ),
  /** `a * b`. */
  mul: binary(
    'mul',
    (a, b) => a * b,
    (_a, b) => b,
    (a) => a,
  ),
  /** `a / b`. */
  div: binary(
    'div',
    (a, b) => a / b,
    (_a, b) => 1 / b,
    (_a, b, y) => -y / b,
  ),
  /** `1 / x`. */
  recip: unary(
    'recip',
    (x) => 1 / x,
    (_x, y) => -y * y,
  ),
  /** `x * x`. */
  square: unary(
    'square',
    (x) => x * x,
    (x) => 2 * x,
  ),
  /** `sqrt(x)`; derivative `+Infinity` at `0`. */
  sqrt: unary(
    'sqrt',
    (x) => Math.sqrt(x),
    (_x, y) => 0.5 / y,
  ),
  /** `e^x`. */
  exp: unary(
    'exp',
    (x) => Math.exp(x),
    (_x, y) => y,
  ),
  /** `e^x - 1`, accurate near `0`. */
  expm1: unary(
    'expm1',
    (x) => Math.expm1(x),
    (x) => Math.exp(x),
  ),
  /** Natural logarithm. */
  log: unary(
    'log',
    (x) => Math.log(x),
    (x) => 1 / x,
  ),
  /** `log(1 + x)`, accurate near `0`. */
  log1p: unary(
    'log1p',
    (x) => Math.log1p(x),
    (x) => 1 / (1 + x),
  ),
  /**
   * `a ** b`, differentiable in both. `d/da = b * a^(b-1)`, taken as `0` when `b === 0` (so
   * `0^0` does not produce `0 * Infinity`). `d/db = a^b * log(a)` for `a > 0` and `0` for
   * `a <= 0`, where the real power is not differentiable in the exponent.
   */
  pow: binary(
    'pow',
    (a, b) => Math.pow(a, b),
    (a, b) => (b === 0 ? 0 : b * Math.pow(a, b - 1)),
    (a, _b, y) => (a > 0 ? y * Math.log(a) : 0),
  ),
  /** `|x|`; derivative `0` at `0`. */
  abs: unary(
    'abs',
    (x) => Math.abs(x),
    (x) => (x > 0 ? 1 : x < 0 ? -1 : x === 0 ? 0 : NaN),
  ),
  /** The smaller of `a` and `b`; on a tie the derivative goes to `a`. `NaN` if either is `NaN`. */
  min: binary(
    'min',
    (a, b) => Math.min(a, b),
    (a, b) => (a <= b ? 1 : 0),
    (a, b) => (a <= b ? 0 : 1),
  ),
  /** The larger of `a` and `b`; on a tie the derivative goes to `a`. `NaN` if either is `NaN`. */
  max: binary(
    'max',
    (a, b) => Math.max(a, b),
    (a, b) => (a >= b ? 1 : 0),
    (a, b) => (a >= b ? 0 : 1),
  ),
  /** Sine. */
  sin: unary(
    'sin',
    (x) => Math.sin(x),
    (x) => Math.cos(x),
  ),
  /** Cosine. */
  cos: unary(
    'cos',
    (x) => Math.cos(x),
    (x) => -Math.sin(x),
  ),
  /** Hyperbolic tangent. */
  tanh: unary(
    'tanh',
    (x) => Math.tanh(x),
    (_x, y) => 1 - y * y,
  ),
  /** Logistic sigmoid `1 / (1 + e^-x)`, stable for any `x`. */
  sigmoid: unary('sigmoid', sigmoidValue, sigmoidDeriv),
  /**
   * Softplus `log(1 + e^x)`, stable for any `x`: computed as `x + log1p(e^-x)` for positive `x`
   * and `log1p(e^x)` otherwise, so it neither overflows nor rounds to `0` for very negative `x`.
   * Its derivative is `sigmoid(x)`.
   */
  softplus: unary(
    'softplus',
    // Stryker disable next-line EqualityOperator: at x = 0 both branches give exactly log(2).
    (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x))),
    (x) => sigmoidValue(x),
  ),
});

/** Names of the one-argument primitives. */
export type UnaryName = {
  [K in keyof typeof primitives]: (typeof primitives)[K] extends { deriv: unknown } ? K : never;
}[keyof typeof primitives];

/** Names of the two-argument primitives. */
export type BinaryName = Exclude<keyof typeof primitives, UnaryName>;

/** The names of the one-argument primitives, in table order. */
export const unaryNames: readonly UnaryName[] = [
  'neg',
  'recip',
  'square',
  'sqrt',
  'exp',
  'expm1',
  'log',
  'log1p',
  'abs',
  'sin',
  'cos',
  'tanh',
  'sigmoid',
  'softplus',
];

/** The names of the two-argument primitives, in table order. */
export const binaryNames: readonly BinaryName[] = ['add', 'sub', 'mul', 'div', 'pow', 'min', 'max'];
