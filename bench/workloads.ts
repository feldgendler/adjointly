/*
 * The functions the benchmark times, each written once against Arithmetic. Inputs are generated
 * from a fixed seed so every run measures the same computation.
 *
 * The library is a parameter rather than an import, so that the A/B comparison (ab.ts) can run
 * exactly these workloads against another checkout's src/. A workload that needs something an
 * older version lacks is left out for that version (see `workloads`), and ab.ts reports it as a
 * case the base does not have.
 */

import type * as Adjointly from '../src/index.ts';

type Arithmetic<T> = Adjointly.Arithmetic<T>;

/** The parts of the library the workloads and the measurement use. */
export type Library = Pick<typeof Adjointly, 'logsumexp' | 'plain' | 'Tape' | 'Forward'>;

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Workload {
  readonly name: string;
  /** The point to evaluate at: the function's inputs. */
  readonly at: readonly number[];
  readonly f: <T>(A: Arithmetic<T>, xs: readonly T[]) => T;
}

/**
 * Logistic loss of a one-hidden-layer tanh network over a synthetic data set: dense mul/add with
 * data entering as constants, the shape of a typical model fit. About 2.2e5 tape nodes.
 */
function network(): Workload {
  const D = 8;
  const H = 8;
  const N = 1000;
  const r = rng(1);
  const X = Float64Array.from({ length: N * D }, () => r() * 2 - 1);
  const Y = Float64Array.from({ length: N }, () => (r() < 0.5 ? 0 : 1));
  const at = Array.from({ length: H * D + H + H + 1 }, () => (r() * 2 - 1) * 0.5);
  function f<T>(A: Arithmetic<T>, p: readonly T[]): T {
    let total = A.constant(0);
    const hidden = new Array<T>(H);
    for (let n = 0; n < N; n++) {
      for (let h = 0; h < H; h++) {
        let z: T = p[H * D + h]!;
        for (let d = 0; d < D; d++) z = A.add(z, A.mul(p[h * D + d]!, A.constant(X[n * D + d]!)));
        hidden[h] = A.tanh(z);
      }
      let o: T = p[H * D + H + H]!;
      for (let h = 0; h < H; h++) o = A.add(o, A.mul(p[H * D + H + h]!, hidden[h]!));
      total = A.add(total, Y[n] === 1 ? A.sub(A.softplus(o), o) : A.softplus(o));
    }
    return A.div(total, A.constant(N));
  }
  return { name: 'network', at, f };
}

/**
 * Many inputs, transcendental-heavy: a log-sum-exp over 2e4 inputs, each passed through
 * sigmoid, log1p and a power. About 1.4e5 tape nodes.
 */
function transcendental(lib: Library): Workload {
  const r = rng(2);
  const at = Array.from({ length: 20000 }, () => r() * 4 - 2);
  function f<T>(A: Arithmetic<T>, xs: readonly T[]): T {
    const half = A.constant(1.5);
    const terms = xs.map((x) => A.pow(A.log1p(A.sigmoid(x)), half));
    return lib.logsumexp(A, terms);
  }
  return { name: 'transcendental', at, f };
}

/**
 * A recurrence driven by data, most of whose operands are fixed coefficients, as in a simulation
 * or a filter: two inputs (an input gain and the initial state) and 20 000 steps. Written twice,
 * with every fixed number a `constant()` node and with the plain-number operations (`mulc`,
 * `subc`), to show what the second spelling saves: the same values and gradient, a third fewer
 * nodes.
 */
function recurrence(numberOperands: boolean): Workload {
  const T = 20000;
  const r = rng(3);
  const u = Float64Array.from({ length: T }, () => r() * 2 - 1);
  const target = Float64Array.from({ length: T }, () => r() * 2 - 1);
  const [w, g, h] = [0.9, 0.7, 0.25];
  const at = [0.8, 0.1]; // input gain, initial state
  function f<T>(A: Arithmetic<T>, [gain, x0]: readonly T[]): T {
    const mulc = numberOperands
      ? (a: T, c: number) => A.mulc(a, c)
      : (a: T, c: number) => A.mul(a, A.constant(c));
    const subc = numberOperands
      ? (a: T, c: number) => A.subc(a, c)
      : (a: T, c: number) => A.sub(a, A.constant(c));
    let x: T = x0!;
    let loss = A.constant(0);
    for (let t = 0; t < T; t++) {
      const drive = A.tanh(A.add(mulc(x, w), mulc(gain!, u[t]!)));
      x = A.add(mulc(drive, g), mulc(x, h));
      loss = A.add(loss, A.square(subc(x, target[t]!)));
    }
    return loss;
  }
  return { name: numberOperands ? 'recurrence-numeric' : 'recurrence-constant', at, f };
}

/** The workloads, built on `lib`; the ones `lib` cannot run are left out. */
export function workloads(lib: Library): readonly Workload[] {
  const hasNumberOperands = (lib.plain as Partial<Arithmetic<number>>).mulc !== undefined;
  return [
    network(),
    transcendental(lib),
    recurrence(false),
    ...(hasNumberOperands ? [recurrence(true)] : []),
  ];
}
