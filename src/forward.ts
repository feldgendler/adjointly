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
 * records. A term whose tangent is exactly zero is skipped, mirroring the tape's rule that a zero
 * adjoint passes nothing on, so an infinite partial such as `sqrt` at `0` does not turn a zero
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

  /** {@inheritDoc Arithmetic.constant} */
  constant(x: number): Dual {
    return this.push(x, 0);
  }

  /** {@inheritDoc Arithmetic.value} */
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
  // monomorphic and inlined; see the same note in tape.ts. neg, add, sub, min and max have finite
  // constant partials, so they need no zero-tangent test: a zero tangent already contributes 0.

  /** {@inheritDoc Arithmetic.neg} */
  neg(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.neg.value(x);
    return this.push(y, P.neg.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.add} */
  add(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.add.value(x, z);
    return this.push(y, P.add.da(x, z, y) * ta + P.add.db(x, z, y) * tb);
  }
  /** {@inheritDoc Arithmetic.sub} */
  sub(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.sub.value(x, z);
    return this.push(y, P.sub.da(x, z, y) * ta + P.sub.db(x, z, y) * tb);
  }
  /** {@inheritDoc Arithmetic.mul} */
  mul(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.mul.value(x, z);
    let t = 0;
    if (ta !== 0) t = P.mul.da(x, z, y) * ta;
    if (tb !== 0) t += P.mul.db(x, z, y) * tb;
    return this.push(y, t);
  }
  /** {@inheritDoc Arithmetic.div} */
  div(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.div.value(x, z);
    let t = 0;
    if (ta !== 0) t = P.div.da(x, z, y) * ta;
    if (tb !== 0) t += P.div.db(x, z, y) * tb;
    return this.push(y, t);
  }
  /** {@inheritDoc Arithmetic.recip} */
  recip(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.recip.value(x);
    return this.push(y, ta === 0 ? 0 : P.recip.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.square} */
  square(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.square.value(x);
    return this.push(y, ta === 0 ? 0 : P.square.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.sqrt} */
  sqrt(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.sqrt.value(x);
    return this.push(y, ta === 0 ? 0 : P.sqrt.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.exp} */
  exp(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.exp.value(x);
    return this.push(y, ta === 0 ? 0 : P.exp.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.expm1} */
  expm1(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.expm1.value(x);
    return this.push(y, ta === 0 ? 0 : P.expm1.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.log} */
  log(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.log.value(x);
    return this.push(y, ta === 0 ? 0 : P.log.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.log1p} */
  log1p(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.log1p.value(x);
    return this.push(y, ta === 0 ? 0 : P.log1p.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.pow} */
  pow(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.pow.value(x, z);
    let t = 0;
    if (ta !== 0) t = P.pow.da(x, z, y) * ta;
    if (tb !== 0) t += P.pow.db(x, z, y) * tb;
    return this.push(y, t);
  }
  /** {@inheritDoc Arithmetic.abs} */
  abs(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.abs.value(x);
    return this.push(y, ta === 0 ? 0 : P.abs.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.min} */
  min(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.min.value(x, z);
    return this.push(y, P.min.da(x, z, y) * ta + P.min.db(x, z, y) * tb);
  }
  /** {@inheritDoc Arithmetic.max} */
  max(a: Dual, b: Dual): Dual {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const ta = this.tan[a]!;
    const tb = this.tan[b]!;
    const y = P.max.value(x, z);
    return this.push(y, P.max.da(x, z, y) * ta + P.max.db(x, z, y) * tb);
  }
  /** {@inheritDoc Arithmetic.sin} */
  sin(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.sin.value(x);
    return this.push(y, ta === 0 ? 0 : P.sin.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.cos} */
  cos(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.cos.value(x);
    return this.push(y, ta === 0 ? 0 : P.cos.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.tanh} */
  tanh(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.tanh.value(x);
    return this.push(y, ta === 0 ? 0 : P.tanh.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.sigmoid} */
  sigmoid(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.sigmoid.value(x);
    return this.push(y, ta === 0 ? 0 : P.sigmoid.deriv(x, y) * ta);
  }
  /** {@inheritDoc Arithmetic.softplus} */
  softplus(a: Dual): Dual {
    const x = this.val[a]!;
    const ta = this.tan[a]!;
    const y = P.softplus.value(x);
    return this.push(y, ta === 0 ? 0 : P.softplus.deriv(x, y) * ta);
  }
}
