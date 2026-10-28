/* Forward mode's storage and tangent rules. */

import { expect, test } from 'vitest';
import { binaryNames, type Dual, Forward, primitives, unaryNames } from '../src/index.ts';
import { expectClose } from './support/util.ts';

test('starts empty with the requested capacity, and grows by doubling', () => {
  expect(new Forward().size).toBe(0);
  for (const capacity of [0, -1, 2.7, NaN, Infinity]) {
    expect(() => new Forward({ capacity }), String(capacity)).toThrow(RangeError);
  }
  expect(() => new Forward({ capacity: NaN })).toThrow(
    'Forward: capacity must be a positive integer, not NaN',
  );
  const F = new Forward({ capacity: 1 });
  const x = F.input(1.0001, 1);
  let y: Dual = x;
  for (let i = 0; i < 1000; i++) y = F.mul(y, x);
  expect(F.size).toBe(1001);
  expectClose(F.value(y), Math.pow(1.0001, 1001), 1e-12);
  expectClose(F.tangent(y), 1001 * Math.pow(1.0001, 1000), 1e-12);
  expect(F.value(x)).toBe(1.0001);
});

test('default capacity holds 1024 values before growing', () => {
  const F = new Forward();
  for (let i = 0; i < 1025; i++) F.input(i, -i);
  expect(F.value(1024 as Dual)).toBe(1024);
  expect(F.tangent(1023 as Dual)).toBe(-1023);
});

test('constants have zero tangent; inputs carry their seed', () => {
  const F = new Forward();
  const c = F.constant(3);
  const x = F.input(2, 0.5);
  expect([F.tangent(c), F.tangent(x)]).toEqual([0, 0.5]);
  const y = F.mul(c, F.square(x)); // 3x^2, d = 6x * 0.5
  expect(F.tangent(y)).toBe(6);
});

test('a zero tangent is never multiplied into an infinite partial', () => {
  const F = new Forward();
  const x = F.input(0, 0); // not moving along this input
  const y = F.input(3, 1);
  const f = F.add(F.sqrt(x), y);
  expect(F.tangent(f)).toBe(1); // not NaN from 0 * Infinity
  const g = F.mul(F.sqrt(x), y); // sqrt(x) has zero tangent, y's partial is sqrt(0) = 0
  expect(F.tangent(g)).toBe(0);
  const h = F.pow(y, F.sqrt(x)); // exponent tangent zero: only d/da contributes, which is 0
  expect(F.tangent(h)).toBe(0);
});

test('both arguments contribute when both move', () => {
  const F = new Forward();
  const a = F.input(2, 1);
  const b = F.input(5, 2);
  expect(F.tangent(F.mul(a, b))).toBe(5 * 1 + 2 * 2);
  expect(F.tangent(F.sub(a, b))).toBe(1 - 2);
});

test('clear forgets the values', () => {
  const F = new Forward();
  F.input(1, 1);
  F.clear();
  expect(F.size).toBe(0);
  expect(F.constant(7)).toBe(0);
});

test('no primitive turns a zero tangent into NaN, at any argument', () => {
  // At these points some partial is infinite or NaN; a zero tangent must still give exactly 0,
  // and a unit tangent on the other argument exactly that argument's partial.
  const specials = [0, -0, 1, -1, Infinity, -Infinity, NaN, 1e308, -1e308];
  // Equal as numbers (so -0 equals 0), with NaN equal to NaN.
  const same = (a: number, b: number): boolean => a === b || (Number.isNaN(a) && Number.isNaN(b));
  for (const name of unaryNames) {
    for (const x of specials) {
      const F = new Forward();
      expect(F.tangent(F[name](F.input(x, 0))) === 0, `${name}(${String(x)})`).toBe(true);
    }
  }
  for (const name of binaryNames) {
    const p = primitives[name];
    for (const a of specials) {
      for (const b of specials) {
        const F = new Forward();
        const y = p.value(a, b);
        const onlyB = F.tangent(F[name](F.input(a, 0), F.input(b, 1)));
        const onlyA = F.tangent(F[name](F.input(a, 1), F.input(b, 0)));
        expect(same(onlyB, p.db(a, b, y)), `${name}(${String(a)}, ${String(b)}) d/db`).toBe(true);
        expect(same(onlyA, p.da(a, b, y)), `${name}(${String(a)}, ${String(b)}) d/da`).toBe(true);
      }
    }
  }
});

test('tangents scale linearly with the seed', () => {
  for (const name of unaryNames) {
    const F = new Forward();
    const x = 0.7;
    const p = primitives[name];
    expect(F.tangent(F[name](F.input(x, 2.5)))).toBe(p.deriv(x, p.value(x)) * 2.5);
  }
  for (const name of binaryNames) {
    const F = new Forward();
    const p = primitives[name];
    const [a, b] = [0.7, 0.4];
    const y = p.value(a, b);
    expect(F.tangent(F[name](F.input(a, 2.5), F.input(b, -3)))).toBe(
      p.da(a, b, y) * 2.5 + p.db(a, b, y) * -3,
    );
  }
});
