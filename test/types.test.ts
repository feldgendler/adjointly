/*
 * Type-level tests. The `@ts-expect-error` lines are verified by `npm run typecheck` (tsc fails if
 * any of them stops being an error); the function holding them is never called.
 */

import { expect, expectTypeOf, test } from 'vitest';
import {
  type Arithmetic,
  type Dual,
  evaluate,
  Forward,
  gradient,
  plain,
  type ScalarFunction,
  Tape,
  type Var,
} from '../src/index.ts';

function typeErrors(): number {
  const t = new Tape();
  const F = new Forward();
  const x = t.input(1);
  const d = F.input(1, 1);

  // @ts-expect-error a raw number is not a Var: numbers enter through constant() or input()
  t.exp(1);
  // @ts-expect-error nor as either argument of a binary operation
  t.add(x, 2);
  // @ts-expect-error a forward-mode Dual is not a tape Var
  t.mul(x, d);
  // @ts-expect-error and a tape Var is not a Dual
  F.sin(x);
  // @ts-expect-error a raw number is not a Dual
  F.tangent(0);
  // @ts-expect-error a plain number is not assignable to Var
  const v: Var = 3;
  // @ts-expect-error a number is not a Mark
  t.rewind(0);

  // A literal point types the inputs as a tuple of its length.
  // @ts-expect-error the point has one input, so there is no y to destructure
  gradient((A, [x, y]) => A.mul(x, y), [3]);
  // @ts-expect-error a function of two inputs cannot be evaluated at a point of one
  evaluate((A: Arithmetic<number>, [x, y]: readonly [number, number]) => A.mul(x, y), [3]);

  // Reading values out is fine: a Var is a number underneath.
  return t.value(x) + v;
}

test('the type errors above are compile-time only', () => {
  expect(typeof typeErrors).toBe('function');
});

test('a literal point gives tuple-typed inputs, and inputs() a tuple of Vars', () => {
  expect(gradient((A, [x, y]) => A.mul(x, y), [3, 4]).gradient).toEqual([4, 3]);
  const [a, b] = new Tape().inputs([1, 2]);
  expectTypeOf(a).toEqualTypeOf<Var>();
  expectTypeOf(b).toEqualTypeOf<Var>();
  expectTypeOf(new Tape().inputs([1, 2] as number[])).toEqualTypeOf<Var[]>();
  expectTypeOf(new Tape().inputs(new Float64Array(2))).toEqualTypeOf<Var[]>();
});

test('the three implementations are Arithmetics of their value types', () => {
  expectTypeOf(plain).toEqualTypeOf<Arithmetic<number>>();
  expectTypeOf(new Tape()).toExtend<Arithmetic<Var>>();
  expectTypeOf(new Forward()).toExtend<Arithmetic<Dual>>();
  // A generic function is a ScalarFunction and accepts any of them.
  const f = <T>(A: Arithmetic<T>, xs: readonly T[]): T => A.exp(xs[0]!);
  expectTypeOf(f).toExtend<ScalarFunction>();
  expect(f(plain, [0])).toBe(1);
});
