/*
 * The functions the benchmark times, each written once against Arithmetic. Inputs are generated
 * from a fixed seed so every run measures the same computation.
 */

import { type Arithmetic, logsumexp } from '../src/index.ts';

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
function transcendental(): Workload {
  const r = rng(2);
  const at = Array.from({ length: 20000 }, () => r() * 4 - 2);
  function f<T>(A: Arithmetic<T>, xs: readonly T[]): T {
    const half = A.constant(1.5);
    const terms = xs.map((x) => A.pow(A.log1p(A.sigmoid(x)), half));
    return logsumexp(A, terms);
  }
  return { name: 'transcendental', at, f };
}

export const workloads: readonly Workload[] = [network(), transcendental()];
