/*
 * Forward mode: {@link Forward}, which carries a tangent alongside every value.
 */

import type { Arithmetic } from './arithmetic.ts';
import { primitives as P } from './primitives.ts';

declare const dualBrand: unique symbol;

/**
 * A value in a {@link Forward} computation: at run time a plain integer, the index of its slot.
 * Branded separately from `Var`, so a forward value cannot be passed to a `Tape` or vice versa, and
 * a raw number cannot be passed where a `Dual` is expected.
 *
 * Valid only on the `Forward` that created it, and only until that `Forward` is cleared.
 */
export type Dual = number & { readonly [dualBrand]: 'adjointly.Dual' };

/**
 * One term of the chain rule, `d * t`, except that a zero factor makes the term zero even when the
 * other factor is infinite or `NaN`. The tape's backward sweep applies the same rule.
 */
function times(d: number, t: number): number {
  return d === 0 || t === 0 ? 0 : d * t;
}

/** Options for constructing a {@link Forward}. */
export interface ForwardOptions {
  /** Initial capacity in values (default 1024); storage doubles whenever it fills. */
  readonly capacity?: number;
}

/**
 * Forward-mode automatic differentiation: an {@link Arithmetic} over {@link Dual} where every value
 * carries a tangent, its derivative along one direction chosen when the inputs are created. One
 * evaluation gives the directional derivative `grad f(x) . v` (a Jacobian-vector product) for every
 * output at once, with no backward sweep and nothing kept but a value and a tangent per slot.
 *
 * Tangents propagate as `t(y) = da * t(a) + db * t(b)` using the same partials the {@link Tape}
 * records, under the same zero rule: if a partial or a tangent is exactly 0 the term is 0, even when
 * the other is infinite or `NaN`, so an infinite partial such as `sqrt` at `0` does not turn a zero
 * tangent into `NaN`.
 *
 * Use forward mode when there are few inputs (or one direction) and many outputs; use the tape
 * when there are many inputs and one output.
 *
 * @example
 * ```ts
 * import { Forward } from 'adjointly';
 * const F = new Forward();
 * const x = F.input(2, 1); // seed: direction (1, 0)
 * const y = F.input(3, 0);
 * const f = F.mul(x, F.sin(y)); // x * sin(y)
 * F.value(f); //=> 2 * sin(3)
 * F.tangent(f); //=> sin(3)
 * ```
 */
export class Forward implements Arithmetic<Dual> {
  private val: Float64Array;
  private tan: Float64Array;
  private n = 0;

  /** An empty forward-mode context; see {@link ForwardOptions}. */
  constructor(options: ForwardOptions = {}) {
    const cap = Math.max(1, Math.floor(options.capacity ?? 1024));
    this.val = new Float64Array(cap);
    this.tan = new Float64Array(cap);
  }

  /** The number of values created so far. */
  get size(): number {
    return this.n;
  }

  /** Forget every value, keeping the storage. Every `Dual` previously issued becomes invalid. */
  clear(): void {
    this.n = 0;
  }

  /** An input with value `x` and tangent (seed) `tangent`: its component of the direction. */
  input(x: number, tangent: number): Dual {
    return this.push(x, tangent);
  }

  /**
   * Lift a number into `T` as a constant: its derivative with respect to every input is zero.
   * On a `Tape` it still appends a node (constants are leaves of the tape).
   */
  constant(x: number): Dual {
    return this.push(x, 0);
  }

  /**
   * The current numeric value of `a`. Use it to branch (`if (A.value(x) > 0)`), to log, or to
   * return a result. The number it returns carries no derivative information: anything computed
   * from it with ordinary arithmetic is a constant as far as differentiation is concerned.
   */
  value(a: Dual): number {
    return this.val[a]!;
  }

  /** The tangent of `a`: its derivative along the direction the inputs were seeded with. */
  tangent(a: Dual): number {
    return this.tan[a]!;
  }

  private push(y: number, t: number): Dual {
    const i = this.n;
    if (i === this.val.length) this.grow();
    this.val[i] = y;
    this.tan[i] = t;
    this.n = i + 1;
    return i as Dual;
  }

  private grow(): void {
    const cap = this.val.length * 2;
    const val = new Float64Array(cap);
    val.set(this.val);
    this.val = val;
    const tan = new Float64Array(cap);
    tan.set(this.tan);
    this.tan = tan;
  }

  // One method per primitive, each calling its own table entry directly, so every call site is
  // monomorphic and inlined; see the same note in tape.ts. neg, add and sub multiply directly:
  // their partials are always 1 or -1, never 0, so `times` would change nothing.

  /** `-a`. */
  neg(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.neg.value(x);
    return this.push(y, P.neg.deriv(x, y) * ta);
  }
  /** `a + b`. */
  add(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.add.value(x, z);
    return this.push(y, P.add.da(x, z, y) * ta + P.add.db(x, z, y) * tb);
  }
  /** `a - b`. */
  sub(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.sub.value(x, z);
    return this.push(y, P.sub.da(x, z, y) * ta + P.sub.db(x, z, y) * tb);
  }
  /** `a * b`. */
  mul(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.mul.value(x, z);
    return this.push(y, times(P.mul.da(x, z, y), ta) + times(P.mul.db(x, z, y), tb));
  }
  /** `a / b`. */
  div(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.div.value(x, z);
    return this.push(y, times(P.div.da(x, z, y), ta) + times(P.div.db(x, z, y), tb));
  }
  /** `1 / a`. */
  recip(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.recip.value(x);
    return this.push(y, times(P.recip.deriv(x, y), ta));
  }
  /** `a * a`, one node instead of a `mul` of `a` with itself. */
  square(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.square.value(x);
    return this.push(y, times(P.square.deriv(x, y), ta));
  }
  /** `sqrt(a)`; its derivative at `0` is `+Infinity`. */
  sqrt(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.sqrt.value(x);
    return this.push(y, times(P.sqrt.deriv(x, y), ta));
  }
  /** `e^a`. */
  exp(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.exp.value(x);
    return this.push(y, times(P.exp.deriv(x, y), ta));
  }
  /** `e^a - 1`, accurate for small `a`. */
  expm1(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.expm1.value(x);
    return this.push(y, times(P.expm1.deriv(x, y), ta));
  }
  /** Natural logarithm. */
  log(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.log.value(x);
    return this.push(y, times(P.log.deriv(x, y), ta));
  }
  /** `log(1 + a)`, accurate for small `a`. */
  log1p(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.log1p.value(x);
    return this.push(y, times(P.log1p.deriv(x, y), ta));
  }
  /**
   * `a ** b`, differentiable in both arguments. The derivative with respect to `b` is `0` wherever
   * `a <= 0`; the derivative with respect to `a` is `0` wherever `b === 0`. For a fixed numeric
   * exponent pass `A.constant(b)`; for `b = 2` prefer {@link Arithmetic.square}, for `b = 0.5`
   * prefer {@link Arithmetic.sqrt}.
   */
  pow(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.pow.value(x, z);
    return this.push(y, times(P.pow.da(x, z, y), ta) + times(P.pow.db(x, z, y), tb));
  }
  /** `|a|`; its derivative at `0` is `0`. */
  abs(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.abs.value(x);
    return this.push(y, times(P.abs.deriv(x, y), ta));
  }
  /** The smaller of `a` and `b`; on a tie the derivative flows to `a`. */
  min(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.min.value(x, z);
    return this.push(y, times(P.min.da(x, z, y), ta) + times(P.min.db(x, z, y), tb));
  }
  /** The larger of `a` and `b`; on a tie the derivative flows to `a`. */
  max(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.max.value(x, z);
    return this.push(y, times(P.max.da(x, z, y), ta) + times(P.max.db(x, z, y), tb));
  }
  /** Sine. */
  sin(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.sin.value(x);
    return this.push(y, times(P.sin.deriv(x, y), ta));
  }
  /** Cosine. */
  cos(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.cos.value(x);
    return this.push(y, times(P.cos.deriv(x, y), ta));
  }
  /** Hyperbolic tangent. */
  tanh(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.tanh.value(x);
    return this.push(y, times(P.tanh.deriv(x, y), ta));
  }
  /** Logistic sigmoid `1 / (1 + e^-a)`, stable for any `a`. */
  sigmoid(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.sigmoid.value(x);
    return this.push(y, times(P.sigmoid.deriv(x, y), ta));
  }
  /** Softplus `log(1 + e^a)`, stable for any `a`; its derivative is `sigmoid(a)`. */
  softplus(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.softplus.value(x);
    return this.push(y, times(P.softplus.deriv(x, y), ta));
  }
}
