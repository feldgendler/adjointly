/*
 * The zero rule: a chain-rule term whose local partial or incoming adjoint/tangent is exactly 0 is
 * 0, even when the other factor is infinite or NaN. Reverse and forward mode both apply it, so they
 * agree where the old rules (a zero adjoint in one mode, a zero tangent in the other) did not.
 */

import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import {
  type Arithmetic,
  type BinaryName,
  binaryNames,
  gradient,
  jvp,
  primitives,
  type ScalarFunction,
  type UnaryName,
  unaryNames,
} from '../src/index.ts';

const both = (f: ScalarFunction, at: number[]): { reverse: number; forward: number } => ({
  reverse: gradient(f, at).gradient[0]!,
  forward: jvp(f, at, [1]).derivative,
});

describe('the zero rule, in both modes', () => {
  test('a zero factor in front of a singular partial', () => {
    // sqrt and log have infinite derivatives at 0; multiplied by 0 the term contributes nothing.
    expect(both((A, [x]) => A.mul(A.constant(0), A.sqrt(x!)), [0])).toEqual({
      reverse: 0,
      forward: 0,
    });
    expect(both((A, [x]) => A.mul(A.constant(0), A.log(x!)), [0])).toEqual({
      reverse: 0,
      forward: 0,
    });
  });

  test('a branch of max that is not taken', () => {
    expect(both((A, [x]) => A.max(A.sqrt(x!), A.constant(1)), [0])).toEqual({
      reverse: 0,
      forward: 0,
    });
  });

  test('a singular partial after a zero partial', () => {
    // 0 * x is 0 for every x, so sqrt(0 * x) and log(0 * x) are constant in x.
    expect(both((A, [x]) => A.sqrt(A.mul(A.constant(0), x!)), [1])).toEqual({
      reverse: 0,
      forward: 0,
    });
    expect(both((A, [x]) => A.log(A.mul(A.constant(0), x!)), [1])).toEqual({
      reverse: 0,
      forward: 0,
    });
  });

  test('a cancellation before a singular partial is the documented limit', () => {
    // Forward mode sees the tangent of x - x cancel to 0; reverse mode adds Infinity - Infinity.
    expect(both((A, [x]) => A.sqrt(A.sub(x!, x!)), [1])).toEqual({ reverse: NaN, forward: 0 });
  });

  test('a nonzero factor still multiplies through', () => {
    expect(both((A, [x]) => A.sqrt(x!), [0])).toEqual({ reverse: Infinity, forward: Infinity });
    expect(both((A, [x]) => A.mul(A.constant(2), A.sqrt(x!)), [0])).toEqual({
      reverse: Infinity,
      forward: Infinity,
    });
  });
});

// Expression trees in which every input appears at most once, so each derivative is the product
// of the partials along one path, which forward mode multiplies from the input up and reverse mode
// from the output down. Two things can still make the groupings differ, and the property excludes
// exactly those (see docs/concepts.md): fan-out, where a cancellation such as x - x is seen by one
// mode before a singular partial and by the other only after it; and a product of finite, nonzero
// partials that underflows to 0 or overflows to Infinity in one grouping but not the other.
type Expr =
  | { readonly k: 'in' }
  | { readonly k: 'const'; readonly v: number }
  | { readonly k: 'un'; readonly op: UnaryName; readonly a: Expr }
  | { readonly k: 'bin'; readonly op: BinaryName; readonly a: Expr; readonly b: Expr };

const { expr } = fc.letrec<{ expr: Expr }>((tie) => ({
  expr: fc.oneof(
    { maxDepth: 5, depthIdentifier: 'zero' },
    fc.constant({ k: 'in' as const }),
    fc.record({ k: fc.constant('const' as const), v: fc.constantFrom(0, -0, 0.5, 1, -2, 3) }),
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

/** Build a tree, numbering its inputs left to right. */
function build<T>(A: Arithmetic<T>, e: Expr, xs: readonly T[], next = { i: 0 }): T {
  switch (e.k) {
    case 'in':
      return xs[next.i++]!;
    case 'const':
      return A.constant(e.v);
    case 'un':
      return A[e.op](build(A, e.a, xs, next));
    case 'bin': {
      const a = build(A, e.a, xs, next);
      return A[e.op](a, build(A, e.b, xs, next));
    }
  }
}

const countInputs = (e: Expr): number =>
  e.k === 'in'
    ? 1
    : e.k === 'const'
      ? 0
      : e.k === 'un'
        ? countInputs(e.a)
        : countInputs(e.a) + countInputs(e.b);

/** For each input, left to right, the partials along its path to the output, input end first. */
function pathPartials(e: Expr, at: readonly number[]): number[][] {
  const paths: number[][] = [];
  const walk = (node: Expr): { value: number; inputs: number[] } => {
    switch (node.k) {
      case 'in':
        paths.push([]);
        return { value: at[paths.length - 1]!, inputs: [paths.length - 1] };
      case 'const':
        return { value: node.v, inputs: [] };
      case 'un': {
        const a = walk(node.a);
        const p = primitives[node.op];
        const y = p.value(a.value);
        const d = p.deriv(a.value, y);
        for (const i of a.inputs) paths[i]!.push(d);
        return { value: y, inputs: a.inputs };
      }
      case 'bin': {
        const a = walk(node.a);
        const b = walk(node.b);
        const p = primitives[node.op];
        const y = p.value(a.value, b.value);
        for (const i of a.inputs) paths[i]!.push(p.da(a.value, b.value, y));
        for (const i of b.inputs) paths[i]!.push(p.db(a.value, b.value, y));
        return { value: y, inputs: [...a.inputs, ...b.inputs] };
      }
    }
  };
  walk(e);
  return paths;
}

const MIN_NORMAL = 2.2250738585072014e-308;

/**
 * Whether multiplying `factors` in order ever leaves the normal range while every factor so far
 * is finite and nonzero: underflow to 0, overflow to Infinity, or a subnormal product, which
 * keeps too few digits for the two groupings to agree to rounding.
 */
function spills(factors: readonly number[]): boolean {
  let product = 1;
  for (const f of factors) {
    if (f === 0 || !Number.isFinite(f)) return false; // from here on both groupings agree
    product *= f;
    if (!(Math.abs(product) >= MIN_NORMAL) || !Number.isFinite(product)) return true;
  }
  return false;
}

const special = fc.constantFrom(
  0,
  -0,
  Infinity,
  -Infinity,
  NaN,
  5e-324,
  -5e-324,
  2.2e-308,
  1,
  -1,
  0.5,
  2,
);

/** The same number, NaN for NaN; or two finite numbers equal to rounding. */
const agree = (a: number, b: number): boolean =>
  Object.is(a, b) ||
  a === b ||
  (Number.isNaN(a) && Number.isNaN(b)) ||
  (Number.isFinite(a) &&
    Number.isFinite(b) &&
    Math.abs(a - b) <= 1e-12 * Math.max(Math.abs(a), Math.abs(b)));

test('reverse and forward mode agree at special points, NaN for NaN', () => {
  fc.assert(
    fc.property(
      expr.filter((e) => countInputs(e) > 0),
      fc.array(special, { minLength: 8, maxLength: 8 }),
      (e, pool) => {
        const at = pool.slice(0, countInputs(e));
        for (const path of pathPartials(e, at)) {
          fc.pre(!spills(path) && !spills([...path].reverse()));
        }
        const f: ScalarFunction = (A, xs) => build(A, e, xs);
        const { gradient: rev } = gradient(f, at);
        for (let i = 0; i < at.length; i++) {
          const dir = at.map((_, j) => (j === i ? 1 : 0));
          const fwd = jvp(f, at, dir).derivative;
          expect(
            agree(rev[i]!, fwd),
            `d/dx${String(i)}: reverse ${String(rev[i])}, forward ${String(fwd)}`,
          ).toBe(true);
        }
      },
    ),
    { numRuns: 2000 },
  );
});
