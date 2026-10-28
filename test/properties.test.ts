/*
 * Property-based tests over random expression trees built from every primitive:
 *
 * - values are bit-for-bit identical under plain, Tape and Forward, for any inputs at all;
 * - reverse mode, forward mode and central finite differences agree, on trees whose operations
 *   are kept inside their domains by construction and away from kinks and overflow by a guard;
 * - a vector-Jacobian product is the weighted sum of single-output gradients.
 */

import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import {
  type Arithmetic,
  type BinaryName,
  binaryNames,
  finiteDifference,
  Forward,
  gradient,
  jvp,
  plain,
  type ScalarFunction,
  Tape,
  type UnaryName,
  unaryNames,
} from '../src/index.ts';

type Expr =
  | { readonly k: 'in'; readonly i: number }
  | { readonly k: 'const'; readonly v: number }
  | { readonly k: 'un'; readonly op: UnaryName; readonly a: Expr }
  | { readonly k: 'bin'; readonly op: BinaryName; readonly a: Expr; readonly b: Expr };

const N_INPUTS = 3;

function exprArbitrary(leafValue: fc.Arbitrary<number>, maxDepth: number): fc.Arbitrary<Expr> {
  const { expr } = fc.letrec<{ expr: Expr }>((tie) => ({
    expr: fc.oneof(
      { maxDepth, depthIdentifier: 'expr' },
      fc.record({ k: fc.constant('in' as const), i: fc.nat(N_INPUTS - 1) }),
      fc.record({ k: fc.constant('const' as const), v: leafValue }),
      fc.record({
        k: fc.constant('un' as const),
        op: fc.constantFrom(...unaryNames),
        a: tie('expr'),
      }),
      fc.record({
        k: fc.constant('bin' as const),
        op: fc.constantFrom(...binaryNames),
        a: tie('expr'),
        b: tie('expr'),
      }),
    ),
  }));
  return expr;
}

/** Build the tree with the primitives exactly as named: any value may come out, NaN included. */
function raw<T>(A: Arithmetic<T>, e: Expr, xs: readonly T[]): T {
  switch (e.k) {
    case 'in':
      return xs[e.i]!;
    case 'const':
      return A.constant(e.v);
    case 'un':
      return A[e.op](raw(A, e.a, xs));
    case 'bin':
      return A[e.op](raw(A, e.a, xs), raw(A, e.b, xs));
  }
}

/**
 * Build the tree keeping every operation inside its domain: arguments that must be positive (or
 * away from zero) are passed through `0.5 + u^2`, and exponents through `tanh`. Every primitive
 * still appears as itself.
 */
function safe<T>(A: Arithmetic<T>, e: Expr, xs: readonly T[]): T {
  const positive = (u: T): T => A.add(A.constant(0.5), A.square(u));
  switch (e.k) {
    case 'in':
      return xs[e.i]!;
    case 'const':
      return A.constant(e.v);
    case 'un': {
      const a = safe(A, e.a, xs);
      switch (e.op) {
        case 'sqrt':
        case 'log':
        case 'recip':
          return A[e.op](positive(a));
        case 'log1p':
          return A.log1p(A.square(a));
        default:
          return A[e.op](a);
      }
    }
    case 'bin': {
      const a = safe(A, e.a, xs);
      const b = safe(A, e.b, xs);
      switch (e.op) {
        case 'div':
          return A.div(a, positive(b));
        case 'pow':
          return A.pow(positive(a), A.mul(A.constant(2), A.tanh(b)));
        default:
          return A[e.op](a, b);
      }
    }
  }
}

/**
 * `plain`, recording whether the evaluation came near trouble for finite differences: a kink of
 * abs/min/max, or a magnitude where exp-of-exp has blown up.
 */
function guarded(): { A: Arithmetic<number>; trouble: () => boolean } {
  let bad = false;
  const watch = (y: number): number => {
    if (!Number.isFinite(y) || Math.abs(y) > 1e4) bad = true;
    return y;
  };
  const A: Arithmetic<number> = { ...plain };
  for (const name of unaryNames) {
    A[name] = (a: number) => watch(plain[name](a));
  }
  for (const name of binaryNames) {
    A[name] = (a: number, b: number) => watch(plain[name](a, b));
  }
  A.abs = (a) => {
    if (Math.abs(a) < 1e-3) bad = true;
    return plain.abs(a);
  };
  A.max = (a, b) => {
    if (Math.abs(a - b) < 1e-3) bad = true;
    return plain.max(a, b);
  };
  A.min = (a, b) => {
    if (Math.abs(a - b) < 1e-3) bad = true;
    return plain.min(a, b);
  };
  return { A, trouble: () => bad };
}

const point = fc.array(fc.double({ min: -3, max: 3, noNaN: true }), {
  minLength: N_INPUTS,
  maxLength: N_INPUTS,
});

const near = (a: number, b: number, rtol: number, atol: number): boolean =>
  Math.abs(a - b) <= atol + rtol * Math.max(Math.abs(a), Math.abs(b));

describe('properties', () => {
  test('plain, Tape and Forward compute bit-identical values for any inputs', () => {
    fc.assert(
      fc.property(
        exprArbitrary(fc.double(), 6),
        fc.array(fc.double(), { minLength: N_INPUTS, maxLength: N_INPUTS }),
        (e, at) => {
          const f: ScalarFunction = (A, xs) => raw(A, e, xs);
          const v = f(plain, at);
          const t = new Tape();
          const tv = t.value(f(t, t.inputs(at)));
          const F = new Forward();
          const fv = F.value(
            f(
              F,
              at.map((x) => F.input(x, 1)),
            ),
          );
          expect(Object.is(tv, v) && Object.is(fv, v)).toBe(true);
        },
      ),
      { numRuns: 500 },
    );
  });

  test('reverse mode equals forward mode along every basis direction', () => {
    fc.assert(
      fc.property(exprArbitrary(fc.double({ min: -2, max: 2, noNaN: true }), 5), point, (e, at) => {
        const f: ScalarFunction = (A, xs) => safe(A, e, xs);
        const g = guarded();
        f(g.A, at);
        fc.pre(!g.trouble());
        const { gradient: rev } = gradient(f, at);
        for (let i = 0; i < N_INPUTS; i++) {
          const dir = at.map((_, j) => (i === j ? 1 : 0));
          const fwd = jvp(f, at, dir).derivative;
          expect(
            near(rev[i]!, fwd, 1e-10, 1e-12),
            `d/dx${String(i)}: ${String(rev[i])} vs ${String(fwd)}`,
          ).toBe(true);
        }
      }),
      { numRuns: 300 },
    );
  });

  test('reverse mode agrees with central finite differences', () => {
    fc.assert(
      fc.property(exprArbitrary(fc.double({ min: -2, max: 2, noNaN: true }), 4), point, (e, at) => {
        const f: ScalarFunction = (A, xs) => safe(A, e, xs);
        const g = guarded();
        f(g.A, at);
        fc.pre(!g.trouble());
        const { gradient: rev } = gradient(f, at);
        const num = finiteDifference(f, at);
        for (let i = 0; i < N_INPUTS; i++) {
          expect(
            near(rev[i]!, num[i]!, 1e-5, 1e-6),
            `d/dx${String(i)}: ${String(rev[i])} vs ${String(num[i])}`,
          ).toBe(true);
        }
      }),
      { numRuns: 300 },
    );
  });

  test('a directional derivative is the gradient dotted with the direction', () => {
    fc.assert(
      fc.property(
        exprArbitrary(fc.double({ min: -2, max: 2, noNaN: true }), 5),
        point,
        point,
        (e, at, dir) => {
          const f: ScalarFunction = (A, xs) => safe(A, e, xs);
          const g = guarded();
          f(g.A, at);
          fc.pre(!g.trouble());
          const rev = gradient(f, at).gradient;
          const want = rev.reduce((s, gi, i) => s + gi * dir[i]!, 0);
          const scale = rev.reduce((s, gi, i) => s + Math.abs(gi * dir[i]!), 0);
          expect(near(jvp(f, at, dir).derivative, want, 1e-10, 1e-12 + 1e-12 * scale)).toBe(true);
        },
      ),
      { numRuns: 200 },
    );
  });

  test('vjp over several outputs equals the weighted sum of their gradients', () => {
    fc.assert(
      fc.property(
        fc.array(exprArbitrary(fc.double({ min: -2, max: 2, noNaN: true }), 4), {
          minLength: 1,
          maxLength: 4,
        }),
        point,
        fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 4, maxLength: 4 }),
        (es, at, weights) => {
          for (const e of es) {
            const g = guarded();
            safe(g.A, e, at);
            fc.pre(!g.trouble());
          }
          const t = new Tape();
          const xs = t.inputs(at);
          const outs = es.map((e) => safe(t, e, xs));
          const w = weights.slice(0, es.length);
          t.vjp(outs, w);
          const got = t.adjoints(xs);
          const want = new Array<number>(N_INPUTS).fill(0);
          const scale = new Array<number>(N_INPUTS).fill(0);
          es.forEach((e, k) => {
            const gk = gradient((A, ys) => safe(A, e, ys), at).gradient;
            gk.forEach((v, i) => {
              want[i]! += w[k]! * v;
              scale[i]! += Math.abs(w[k]! * v);
            });
          });
          for (let i = 0; i < N_INPUTS; i++) {
            expect(near(got[i]!, want[i]!, 1e-10, 1e-12 + 1e-12 * scale[i]!)).toBe(true);
          }
        },
      ),
      { numRuns: 200 },
    );
  });
});
