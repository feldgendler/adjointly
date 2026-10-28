/*
 * Reverse mode: the {@link Tape}, which records a computation into typed arrays and sweeps it
 * backwards for gradients.
 */

import type { Arithmetic } from './arithmetic.ts';
import { primitives as P } from './primitives.ts';

declare const varBrand: unique symbol;
declare const markBrand: unique symbol;

/**
 * A variable on a {@link Tape}: at run time a plain integer, the index of its node. The brand makes
 * it a compile-time error to pass a raw number where a `Var` is expected, or to use a `Var` from a
 * {@link Forward} where a tape's is expected; numbers enter a tape through
 * {@link Tape.input} or {@link Tape.constant}.
 *
 * A `Var` is only meaningful on the tape that created it, and only until that tape is cleared or
 * rewound past it. Neither is checked at run time.
 *
 * Because it is a number underneath, `x + y` on two `Var`s type-checks (TypeScript allows
 * arithmetic on number subtypes) and silently adds two indices. Always use `tape.add(x, y)`.
 */
export type Var = number & { readonly [varBrand]: 'adjointly.Var' };

/** A checkpoint on a {@link Tape}, returned by {@link Tape.mark} and consumed by {@link Tape.rewind}. */
export type Mark = number & { readonly [markBrand]: 'adjointly.Mark' };

/** Options for constructing a {@link Tape}. */
export interface TapeOptions {
  /**
   * Initial capacity in nodes (default 1024). The tape doubles whenever it fills, so this only
   * matters to avoid the first few reallocations when the size of the computation is known.
   */
  readonly capacity?: number;
}

/**
 * Slot 0 of every array is a sink, not a node: leaves name it as both parents, and unary nodes as
 * their second, with a zero partial. The backward sweep can then add into both parents of every
 * node without testing for a missing one; whatever lands in the sink is never read.
 */
const SINK = 0;

/**
 * Reverse-mode automatic differentiation: an {@link Arithmetic} over {@link Var} that records
 * every operation as a node on a tape, then computes the derivative of one output with respect to
 * every node in a single backward sweep.
 *
 * Each node stores its value, up to two parent indices and the local partial derivatives with
 * respect to them, computed when the node is recorded. The storage is five typed arrays that
 * double when full, so recording does not allocate once the tape has grown to the size of the
 * computation, and a tape reused across evaluations (via {@link Tape.clear} or
 * {@link Tape.rewind}) never allocates again.
 *
 * The backward sweep visits nodes from the output down to the first, adding
 * `partial * adjoint(child)` into each parent's adjoint. A node whose adjoint is exactly zero
 * passes nothing on, even if its partials are infinite or `NaN`, so an unused `sqrt(0)` or
 * `log(0)` does not poison the gradient.
 *
 * @example
 * ```ts
 * import { Tape } from 'adjointly';
 * const t = new Tape();
 * const x = t.input(2);
 * const y = t.input(3);
 * const f = t.mul(x, t.sin(y)); // x * sin(y)
 * t.backward(f);
 * t.adjoint(x); //=> sin(3)
 * t.adjoint(y); //=> 2 * cos(3)
 * ```
 */
export class Tape implements Arithmetic<Var> {
  private val: Float64Array;
  private pa: Int32Array;
  private pb: Int32Array;
  private da: Float64Array;
  private db: Float64Array;
  private adj: Float64Array;
  /** The next free slot; nodes occupy slots 1 to n - 1. */
  private n = 1;

  /** An empty tape; see {@link TapeOptions}. */
  constructor(options: TapeOptions = {}) {
    const cap = Math.max(1, Math.floor(options.capacity ?? 1024)) + 1; // + the sink
    this.val = new Float64Array(cap);
    this.pa = new Int32Array(cap);
    this.pb = new Int32Array(cap);
    this.da = new Float64Array(cap);
    this.db = new Float64Array(cap);
    this.adj = new Float64Array(cap);
  }

  /** The number of nodes recorded so far. */
  get size(): number {
    return this.n - 1;
  }

  /** The number of nodes the tape can hold before it next grows. */
  get capacity(): number {
    return this.val.length - 1;
  }

  /** Forget every node, keeping the storage. Every `Var` previously issued becomes invalid. */
  clear(): void {
    this.n = 1;
  }

  /**
   * A checkpoint at the current end of the tape. {@link Tape.rewind} to it discards everything
   * recorded after it and keeps everything before, so a shared prefix (inputs, parameters,
   * precomputed features) is recorded once and reused across many evaluations.
   *
   * @example
   * ```ts
   * const t = new Tape();
   * const w = t.input(0.5);
   * const m = t.mark();
   * let grad = 0;
   * for (const x of [1, 2, 3]) {
   *   t.rewind(m); // drop the previous sample's nodes, keep w
   *   const loss = t.square(t.sub(t.mul(w, t.constant(x)), t.constant(1)));
   *   t.backward(loss);
   *   grad += t.adjoint(w);
   * }
   * grad; //=> 2 * ((0.5 - 1) * 1 + (1 - 1) * 2 + (1.5 - 1) * 3)
   * t.size; //=> 6
   * ```
   */
  mark(): Mark {
    return this.n as Mark;
  }

  /**
   * Truncate the tape back to `mark`. Vars created before the mark stay valid with their values;
   * Vars created after it become invalid. Throws if `mark` is beyond the current end (the tape was
   * already cleared or rewound further back).
   */
  rewind(mark: Mark): void {
    if (mark > this.n || mark < 1) {
      throw new RangeError(`Tape.rewind: mark ${mark} is outside the tape (size ${this.size})`);
    }
    this.n = mark;
  }

  /**
   * Record an independent variable with value `x`. Identical to {@link Tape.constant} on the tape
   * (both are leaves and both accumulate an adjoint); the separate name says which leaves you
   * intend to read a gradient for.
   */
  input(x: number): Var {
    return this.push(x, SINK, 0, SINK, 0);
  }

  /** Record several independent variables at once; see {@link Tape.input}. */
  inputs(xs: ArrayLike<number>): Var[] {
    return Array.from(xs, (x) => this.push(x, SINK, 0, SINK, 0));
  }

  /** {@inheritDoc Arithmetic.constant} */
  constant(x: number): Var {
    return this.push(x, SINK, 0, SINK, 0);
  }

  /** {@inheritDoc Arithmetic.value} */
  value(a: Var): number {
    return this.val[a]!;
  }

  // ---- backward ----

  /**
   * Run one backward sweep from `output`, seeding its adjoint with `seed` (default 1). Afterwards
   * {@link Tape.adjoint} of any node `v` is `seed * d(output)/d(v)`. Every call starts from zero
   * adjoints, so call it once per output whose gradient you want; the recorded tape is not
   * changed and can be swept any number of times.
   */
  backward(output: Var, seed = 1): void {
    if (output < 1 || output >= this.n) {
      throw new RangeError(
        `Tape.backward: ${output} is not a node of this tape (size ${this.size})`,
      );
    }
    const adj = this.adj;
    adj.fill(0, 0, output + 1);
    adj[output] = seed;
    this.sweep(output);
  }

  /**
   * Vector-Jacobian product: one backward sweep seeded with `weights[k]` at `outputs[k]`.
   * Afterwards {@link Tape.adjoint} of `v` is `sum_k weights[k] * d(outputs[k])/d(v)`. An output
   * listed twice gets the sum of its weights; with no outputs every adjoint is zero.
   *
   * @example
   * ```ts
   * const t = new Tape();
   * const x = t.input(1);
   * const f = t.square(x); // df/dx = 2x
   * const g = t.exp(x); // dg/dx = e^x
   * t.vjp([f, g], [2, 3]); // gradient of 2f + 3g in one sweep
   * t.adjoint(x); //=> 2 * 2 + 3 * E
   * ```
   */
  vjp(outputs: readonly Var[], weights: ArrayLike<number>): void {
    if (outputs.length !== weights.length) {
      throw new RangeError(`Tape.vjp: ${outputs.length} outputs but ${weights.length} weights`);
    }
    for (const o of outputs) {
      if (o < 1 || o >= this.n) {
        throw new RangeError(`Tape.vjp: ${o} is not a node of this tape (size ${this.size})`);
      }
    }
    const adj = this.adj;
    if (outputs.length === 0) {
      adj.fill(0);
      return;
    }
    const top = Math.max(...outputs); // sweep from the latest output
    adj.fill(0, 0, top + 1);
    outputs.forEach((o, k) => {
      adj[o] = adj[o]! + weights[k]!;
    });
    this.sweep(top);
  }

  /**
   * The adjoint of `v` from the last {@link Tape.backward} or {@link Tape.vjp}: the derivative of
   * the seeded output(s) with respect to `v`. Zero for a node the output does not depend on.
   * Meaningless for a node recorded after the output of that sweep.
   */
  adjoint(v: Var): number {
    return this.adj[v]!;
  }

  /**
   * Read the adjoints of `vars` into `out` (allocated if not given) and return it: the gradient
   * restricted to those variables, in their order.
   */
  adjoints(vars: readonly Var[], out: Float64Array = new Float64Array(vars.length)): Float64Array {
    const adj = this.adj;
    vars.forEach((v, k) => {
      out[k] = adj[v]!;
    });
    return out;
  }

  /**
   * `backward(output)` then `adjoints(inputs, out)`: the gradient of `output` with respect to
   * `inputs`, in their order.
   *
   * @example
   * ```ts
   * const t = new Tape();
   * const [x, y] = t.inputs([1, 2]) as [Var, Var];
   * t.gradient(t.mul(x, y), [x, y]); //=> [2, 1]
   * ```
   */
  gradient(output: Var, inputs: readonly Var[], out?: Float64Array): Float64Array {
    this.backward(output);
    return this.adjoints(inputs, out);
  }

  private sweep(top: number): void {
    const adj = this.adj,
      pa = this.pa,
      pb = this.pb,
      da = this.da,
      db = this.db;
    // Leaves (inputs and constants) are skipped although adding their zero partials into the sink
    // would be harmless: every leaf would read and write the same slot, a chain of dependent
    // memory operations that made the whole gradient about 25% slower on a constant-heavy loss.
    // Both that test and visiting the sink itself (i >= 0) change no result, only the speed.
    // Stryker disable next-line EqualityOperator: also visiting the sink changes nothing read.
    for (let i = top; i > SINK; i--) {
      const g = adj[i]!;
      if (g === 0) continue;
      const a = pa[i]!;
      // Stryker disable next-line ConditionalExpression,EqualityOperator: skipping leaves is for speed only.
      if (a === SINK) continue;
      const b = pb[i]!;
      adj[a] = adj[a]! + da[i]! * g;
      adj[b] = adj[b]! + db[i]! * g;
    }
  }

  // ---- recording ----

  private push(y: number, a: number, dA: number, b: number, dB: number): Var {
    const i = this.n;
    if (i === this.val.length) this.grow();
    this.val[i] = y;
    this.pa[i] = a;
    this.da[i] = dA;
    this.pb[i] = b;
    this.db[i] = dB;
    this.n = i + 1;
    return i as Var;
  }

  private grow(): void {
    const cap = this.val.length * 2;
    const val = new Float64Array(cap);
    val.set(this.val);
    this.val = val;
    const pa = new Int32Array(cap);
    pa.set(this.pa);
    this.pa = pa;
    const pb = new Int32Array(cap);
    pb.set(this.pb);
    this.pb = pb;
    const da = new Float64Array(cap);
    da.set(this.da);
    this.da = da;
    const db = new Float64Array(cap);
    db.set(this.db);
    this.db = db;
    const adj = new Float64Array(cap);
    adj.set(this.adj); // adjoints of the last sweep stay readable while recording continues
    this.adj = adj;
  }

  // One method per primitive, each calling its own table entry directly. A shared helper taking
  // the table entry as an argument was 2x slower: its call sites see every primitive, so V8 cannot
  // inline them, while these are monomorphic. Keep the table the only place a formula is written.

  /** {@inheritDoc Arithmetic.neg} */
  neg(a: Var): Var {
    const x = this.val[a]!;
    const y = P.neg.value(x);
    return this.push(y, a, P.neg.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.add} */
  add(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.add.value(x, z);
    return this.push(y, a, P.add.da(x, z, y), b, P.add.db(x, z, y));
  }
  /** {@inheritDoc Arithmetic.sub} */
  sub(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.sub.value(x, z);
    return this.push(y, a, P.sub.da(x, z, y), b, P.sub.db(x, z, y));
  }
  /** {@inheritDoc Arithmetic.mul} */
  mul(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.mul.value(x, z);
    return this.push(y, a, P.mul.da(x, z, y), b, P.mul.db(x, z, y));
  }
  /** {@inheritDoc Arithmetic.div} */
  div(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.div.value(x, z);
    return this.push(y, a, P.div.da(x, z, y), b, P.div.db(x, z, y));
  }
  /** {@inheritDoc Arithmetic.recip} */
  recip(a: Var): Var {
    const x = this.val[a]!;
    const y = P.recip.value(x);
    return this.push(y, a, P.recip.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.square} */
  square(a: Var): Var {
    const x = this.val[a]!;
    const y = P.square.value(x);
    return this.push(y, a, P.square.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.sqrt} */
  sqrt(a: Var): Var {
    const x = this.val[a]!;
    const y = P.sqrt.value(x);
    return this.push(y, a, P.sqrt.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.exp} */
  exp(a: Var): Var {
    const x = this.val[a]!;
    const y = P.exp.value(x);
    return this.push(y, a, P.exp.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.expm1} */
  expm1(a: Var): Var {
    const x = this.val[a]!;
    const y = P.expm1.value(x);
    return this.push(y, a, P.expm1.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.log} */
  log(a: Var): Var {
    const x = this.val[a]!;
    const y = P.log.value(x);
    return this.push(y, a, P.log.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.log1p} */
  log1p(a: Var): Var {
    const x = this.val[a]!;
    const y = P.log1p.value(x);
    return this.push(y, a, P.log1p.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.pow} */
  pow(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.pow.value(x, z);
    return this.push(y, a, P.pow.da(x, z, y), b, P.pow.db(x, z, y));
  }
  /** {@inheritDoc Arithmetic.abs} */
  abs(a: Var): Var {
    const x = this.val[a]!;
    const y = P.abs.value(x);
    return this.push(y, a, P.abs.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.min} */
  min(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.min.value(x, z);
    return this.push(y, a, P.min.da(x, z, y), b, P.min.db(x, z, y));
  }
  /** {@inheritDoc Arithmetic.max} */
  max(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.max.value(x, z);
    return this.push(y, a, P.max.da(x, z, y), b, P.max.db(x, z, y));
  }
  /** {@inheritDoc Arithmetic.sin} */
  sin(a: Var): Var {
    const x = this.val[a]!;
    const y = P.sin.value(x);
    return this.push(y, a, P.sin.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.cos} */
  cos(a: Var): Var {
    const x = this.val[a]!;
    const y = P.cos.value(x);
    return this.push(y, a, P.cos.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.tanh} */
  tanh(a: Var): Var {
    const x = this.val[a]!;
    const y = P.tanh.value(x);
    return this.push(y, a, P.tanh.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.sigmoid} */
  sigmoid(a: Var): Var {
    const x = this.val[a]!;
    const y = P.sigmoid.value(x);
    return this.push(y, a, P.sigmoid.deriv(x, y), SINK, 0);
  }
  /** {@inheritDoc Arithmetic.softplus} */
  softplus(a: Var): Var {
    const x = this.val[a]!;
    const y = P.softplus.value(x);
    return this.push(y, a, P.softplus.deriv(x, y), SINK, 0);
  }
}
