/*
 * Reverse mode: the {@link Tape}, which records a computation into typed arrays and sweeps it
 * backwards for gradients.
 */

import type { Arithmetic } from './arithmetic.ts';
import { initialCapacity } from './capacity.ts';
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
   * Initial capacity in nodes, a positive integer (default 1024; anything else throws a
   * `RangeError`). The tape doubles whenever it fills, so this only matters to avoid the first few
   * reallocations when the size of the computation is known.
   */
  readonly capacity?: number;
}

/**
 * Slot 0 of every array is a sink, not a node: leaves name it as both parents, and unary nodes as
 * their second, always with a zero partial. Every parent index is therefore a valid slot, and the
 * backward sweep's zero rule never writes to the sink.
 */
const SINK = 0;

/**
 * Reverse-mode automatic differentiation: an {@link Arithmetic} over {@link Var} that records
 * every operation as a node on a tape, then computes the derivative of one output with respect to
 * every node in a single backward sweep.
 *
 * Each node stores its value, up to two parent indices and the local partial derivatives with
 * respect to them, computed when the node is recorded, and its adjoint: six typed arrays, 40
 * bytes a node. They double when full, so recording does not allocate once the tape has grown to
 * the size of the computation, and a tape reused across evaluations (via {@link Tape.clear} or
 * {@link Tape.rewind}) never allocates again.
 *
 * The backward sweep visits nodes from the output down to the first, adding
 * `partial * adjoint(child)` into each parent's adjoint, under the zero rule: if the partial or
 * the adjoint is exactly 0 the term is 0, even when the other is infinite or `NaN`, so an unused
 * `sqrt(0)` or `log(0)` does not poison the gradient.
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
    const cap = initialCapacity(options.capacity, 'Tape') + 1; // + the sink
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
   *   const loss = t.square(t.subc(t.mulc(w, x), 1));
   *   t.backward(loss);
   *   grad += t.adjoint(w);
   * }
   * grad; //=> 2 * ((0.5 - 1) * 1 + (1 - 1) * 2 + (1.5 - 1) * 3)
   * t.size; //=> 4
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

  /**
   * Record several independent variables at once; see {@link Tape.input}. For a tuple of numbers
   * the result is a tuple of the same length, so it destructures without non-null assertions.
   *
   * @example
   * ```ts
   * const t = new Tape();
   * const [x, y] = t.inputs([2, 3]);
   * t.value(t.mul(x, y)); //=> 6
   * ```
   */
  inputs<const X extends readonly number[]>(xs: X): { -readonly [K in keyof X]: Var };
  inputs(xs: ArrayLike<number>): Var[];
  inputs(xs: ArrayLike<number>): Var[] {
    return Array.from(xs, (x) => this.push(x, SINK, 0, SINK, 0));
  }

  /**
   * Lift a number into `T` as a constant: its derivative with respect to every input is zero.
   * On a `Tape` it still appends a node (constants are leaves of the tape).
   */
  constant(x: number): Var {
    return this.push(x, SINK, 0, SINK, 0);
  }

  /**
   * The current numeric value of `a`. Use it to branch (`if (A.value(x) > 0)`), to log, or to
   * return a result. The number it returns carries no derivative information: anything computed
   * from it with ordinary arithmetic is a constant as far as differentiation is concerned.
   */
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
    let top = 0; // the latest output, where the sweep starts
    for (const o of outputs) {
      if (o < 1 || o >= this.n) {
        throw new RangeError(`Tape.vjp: ${o} is not a node of this tape (size ${this.size})`);
      }
      top = Math.max(top, o);
    }
    const adj = this.adj;
    if (outputs.length === 0) {
      adj.fill(0);
      return;
    }
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
   * const [x, y] = t.inputs([1, 2]);
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
    // A leaf's partials are 0 and so is a unary node's second, so the zero rule below also keeps
    // every write out of the sink.
    // Stryker disable next-line EqualityOperator: also visiting the sink changes nothing read.
    for (let i = top; i > SINK; i--) {
      // 0 times anything is 0: a zero adjoint or a zero partial passes nothing on, even where the
      // other factor is infinite or NaN. Forward mode applies the same rule (`times` there).
      const g = adj[i]!;
      if (g === 0) continue;
      const dA = da[i]!;
      if (dA !== 0) {
        const a = pa[i]!;
        adj[a] = adj[a]! + dA * g;
      }
      const dB = db[i]!;
      if (dB !== 0) {
        const b = pb[i]!;
        adj[b] = adj[b]! + dB * g;
      }
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

  /** `-a`. */
  neg(a: Var): Var {
    const x = this.val[a]!;
    const y = P.neg.value(x);
    return this.push(y, a, P.neg.deriv(x, y), SINK, 0);
  }
  /** `a + b`. */
  add(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.add.value(x, z);
    return this.push(y, a, P.add.da(x, z, y), b, P.add.db(x, z, y));
  }
  /** `a - b`. */
  sub(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.sub.value(x, z);
    return this.push(y, a, P.sub.da(x, z, y), b, P.sub.db(x, z, y));
  }
  /** `a * b`. */
  mul(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.mul.value(x, z);
    return this.push(y, a, P.mul.da(x, z, y), b, P.mul.db(x, z, y));
  }
  /** `a / b`. */
  div(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.div.value(x, z);
    return this.push(y, a, P.div.da(x, z, y), b, P.div.db(x, z, y));
  }
  /** `1 / a`. */
  recip(a: Var): Var {
    const x = this.val[a]!;
    const y = P.recip.value(x);
    return this.push(y, a, P.recip.deriv(x, y), SINK, 0);
  }
  /** `a * a`, one node instead of a `mul` of `a` with itself. */
  square(a: Var): Var {
    const x = this.val[a]!;
    const y = P.square.value(x);
    return this.push(y, a, P.square.deriv(x, y), SINK, 0);
  }
  /** `sqrt(a)`; its derivative at `0` is `+Infinity`. */
  sqrt(a: Var): Var {
    const x = this.val[a]!;
    const y = P.sqrt.value(x);
    return this.push(y, a, P.sqrt.deriv(x, y), SINK, 0);
  }
  /** `e^a`. */
  exp(a: Var): Var {
    const x = this.val[a]!;
    const y = P.exp.value(x);
    return this.push(y, a, P.exp.deriv(x, y), SINK, 0);
  }
  /** `e^a - 1`, accurate for small `a`. */
  expm1(a: Var): Var {
    const x = this.val[a]!;
    const y = P.expm1.value(x);
    return this.push(y, a, P.expm1.deriv(x, y), SINK, 0);
  }
  /** Natural logarithm. */
  log(a: Var): Var {
    const x = this.val[a]!;
    const y = P.log.value(x);
    return this.push(y, a, P.log.deriv(x, y), SINK, 0);
  }
  /** `log(1 + a)`, accurate for small `a`. */
  log1p(a: Var): Var {
    const x = this.val[a]!;
    const y = P.log1p.value(x);
    return this.push(y, a, P.log1p.deriv(x, y), SINK, 0);
  }
  /**
   * `a ** b`, differentiable in both arguments. The derivative with respect to `b` is `0` wherever
   * `a <= 0`; the derivative with respect to `a` is `0` wherever `b === 0`. For a fixed numeric
   * exponent use {@link Arithmetic.powc}; for `b = 2` prefer {@link Arithmetic.square}, for
   * `b = 0.5` prefer {@link Arithmetic.sqrt}.
   */
  pow(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.pow.value(x, z);
    return this.push(y, a, P.pow.da(x, z, y), b, P.pow.db(x, z, y));
  }
  /** `|a|`; its derivative at `0` is `0`. */
  abs(a: Var): Var {
    const x = this.val[a]!;
    const y = P.abs.value(x);
    return this.push(y, a, P.abs.deriv(x, y), SINK, 0);
  }
  /** The smaller of `a` and `b`; on a tie the derivative flows to `a`. */
  min(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.min.value(x, z);
    return this.push(y, a, P.min.da(x, z, y), b, P.min.db(x, z, y));
  }
  /** The larger of `a` and `b`; on a tie the derivative flows to `a`. */
  max(a: Var, b: Var): Var {
    const x = this.val[a]!;
    const z = this.val[b]!;
    const y = P.max.value(x, z);
    return this.push(y, a, P.max.da(x, z, y), b, P.max.db(x, z, y));
  }
  /** Sine. */
  sin(a: Var): Var {
    const x = this.val[a]!;
    const y = P.sin.value(x);
    return this.push(y, a, P.sin.deriv(x, y), SINK, 0);
  }
  /** Cosine. */
  cos(a: Var): Var {
    const x = this.val[a]!;
    const y = P.cos.value(x);
    return this.push(y, a, P.cos.deriv(x, y), SINK, 0);
  }
  /** Hyperbolic tangent. */
  tanh(a: Var): Var {
    const x = this.val[a]!;
    const y = P.tanh.value(x);
    return this.push(y, a, P.tanh.deriv(x, y), SINK, 0);
  }
  /** Logistic sigmoid `1 / (1 + e^-a)`, stable for any `a`. */
  sigmoid(a: Var): Var {
    const x = this.val[a]!;
    const y = P.sigmoid.value(x);
    return this.push(y, a, P.sigmoid.deriv(x, y), SINK, 0);
  }
  /** Softplus `log(1 + e^a)`, stable for any `a`; its derivative is `sigmoid(a)`. */
  softplus(a: Var): Var {
    const x = this.val[a]!;
    const y = P.softplus.value(x);
    return this.push(y, a, P.softplus.deriv(x, y), SINK, 0);
  }

  // The constant-operand operations: the binary primitive with one operand a plain number, one
  // node with one parent.

  /**
   * `a + c` for a plain number `c`: `add(a, constant(c))` without the constant's tape node. Same
   * value and derivative, bit for bit.
   */
  addc(a: Var, c: number): Var {
    const x = this.val[a]!;
    const y = P.add.value(x, c);
    return this.push(y, a, P.add.da(x, c, y), SINK, 0);
  }

  /** `a - c` for a plain number `c`; `sub(a, constant(c))` in one node. */
  subc(a: Var, c: number): Var {
    const x = this.val[a]!;
    const y = P.sub.value(x, c);
    return this.push(y, a, P.sub.da(x, c, y), SINK, 0);
  }

  /** `c - a` for a plain number `c`; `sub(constant(c), a)` in one node. */
  csub(c: number, a: Var): Var {
    const x = this.val[a]!;
    const y = P.sub.value(c, x);
    return this.push(y, a, P.sub.db(c, x, y), SINK, 0);
  }

  /** `a * c` for a plain number `c`; `mul(a, constant(c))` in one node. */
  mulc(a: Var, c: number): Var {
    const x = this.val[a]!;
    const y = P.mul.value(x, c);
    return this.push(y, a, P.mul.da(x, c, y), SINK, 0);
  }

  /** `a / c` for a plain number `c`; `div(a, constant(c))` in one node. */
  divc(a: Var, c: number): Var {
    const x = this.val[a]!;
    const y = P.div.value(x, c);
    return this.push(y, a, P.div.da(x, c, y), SINK, 0);
  }

  /** `c / a` for a plain number `c`; `div(constant(c), a)` in one node. */
  cdiv(c: number, a: Var): Var {
    const x = this.val[a]!;
    const y = P.div.value(c, x);
    return this.push(y, a, P.div.db(c, x, y), SINK, 0);
  }

  /**
   * `a ** c` for a plain number `c`; `pow(a, constant(c))` in one node, with the same conventions
   * (the derivative is `0` when `c === 0`).
   */
  powc(a: Var, c: number): Var {
    const x = this.val[a]!;
    const y = P.pow.value(x, c);
    return this.push(y, a, P.pow.da(x, c, y), SINK, 0);
  }
}
