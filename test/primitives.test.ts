/*
 * Every primitive's value and derivative, checked against independently written analytic
 * formulas, through the table itself and through all three implementations of Arithmetic.
 */

import { describe, expect, test } from 'vitest';
import {
  type BinaryName,
  binaryNames,
  Forward,
  plain,
  primitives,
  Tape,
  type UnaryName,
  unaryNames,
} from '../src/index.ts';
import { expectClose } from './support/util.ts';

interface UnaryCase {
  readonly f: (x: number) => number;
  readonly df: (x: number) => number;
  readonly at: readonly number[];
}

const GENERAL = [-7.5, -2, -1, -0.3, 1e-3, 0.3, 1, 2, 7.5];
const POSITIVE = [1e-3, 0.3, 1, 2, 7.5, 1e3];

const unary: Record<UnaryName, UnaryCase> = {
  neg: { f: (x) => -x, df: () => -1, at: GENERAL },
  recip: { f: (x) => 1 / x, df: (x) => -1 / (x * x), at: GENERAL },
  square: { f: (x) => x ** 2, df: (x) => 2 * x, at: GENERAL },
  sqrt: { f: Math.sqrt, df: (x) => 1 / (2 * Math.sqrt(x)), at: POSITIVE },
  exp: { f: Math.exp, df: Math.exp, at: GENERAL },
  expm1: { f: (x) => Math.exp(x) - 1, df: Math.exp, at: GENERAL },
  log: { f: Math.log, df: (x) => 1 / x, at: POSITIVE },
  log1p: { f: (x) => Math.log(1 + x), df: (x) => 1 / (1 + x), at: [-0.9, ...POSITIVE] },
  abs: { f: (x) => (x < 0 ? -x : x), df: (x) => (x < 0 ? -1 : 1), at: GENERAL },
  sin: { f: Math.sin, df: Math.cos, at: GENERAL },
  cos: { f: Math.cos, df: (x) => -Math.sin(x), at: GENERAL },
  tanh: { f: Math.tanh, df: (x) => 1 / Math.cosh(x) ** 2, at: GENERAL },
  sigmoid: {
    f: (x) => 1 / (1 + Math.exp(-x)),
    df: (x) => Math.exp(-x) / (1 + Math.exp(-x)) ** 2,
    at: GENERAL,
  },
  softplus: { f: (x) => Math.log(1 + Math.exp(x)), df: (x) => 1 / (1 + Math.exp(-x)), at: GENERAL },
};

interface BinaryCase {
  readonly f: (a: number, b: number) => number;
  readonly da: (a: number, b: number) => number;
  readonly db: (a: number, b: number) => number;
  readonly at: readonly (readonly [number, number])[];
}

const PAIRS = [
  [-2, 3],
  [0.5, -1.5],
  [3, 0.25],
  [-4, -0.5],
  [7, 2],
] as const;
const POSITIVE_BASE = [
  [0.5, -1.5],
  [3, 0.25],
  [2, 3],
  [7, -2],
  [1, 4],
] as const;

const binary: Record<BinaryName, BinaryCase> = {
  add: { f: (a, b) => a + b, da: () => 1, db: () => 1, at: PAIRS },
  sub: { f: (a, b) => a - b, da: () => 1, db: () => -1, at: PAIRS },
  mul: { f: (a, b) => a * b, da: (_a, b) => b, db: (a) => a, at: PAIRS },
  div: { f: (a, b) => a / b, da: (_a, b) => 1 / b, db: (a, b) => -a / (b * b), at: PAIRS },
  pow: {
    f: (a, b) => Math.exp(b * Math.log(a)),
    da: (a, b) => b * Math.exp((b - 1) * Math.log(a)),
    db: (a, b) => Math.exp(b * Math.log(a)) * Math.log(a),
    at: POSITIVE_BASE,
  },
  min: {
    f: (a, b) => (a < b ? a : b),
    da: (a, b) => (a < b ? 1 : 0),
    db: (a, b) => (a < b ? 0 : 1),
    at: PAIRS,
  },
  max: {
    f: (a, b) => (a > b ? a : b),
    da: (a, b) => (a > b ? 1 : 0),
    db: (a, b) => (a > b ? 0 : 1),
    at: PAIRS,
  },
};

/** Value and derivative of a unary primitive through each implementation. */
function unaryEverywhere(name: UnaryName, x: number) {
  const p = primitives[name];
  const tableValue = p.value(x);
  const tableDeriv = p.deriv(x, tableValue);

  const t = new Tape();
  const xv = t.input(x);
  const y = t[name](xv);
  t.backward(y);

  const F = new Forward();
  const yf = F[name](F.input(x, 1));

  return {
    tableValue,
    tableDeriv,
    plainValue: plain[name](x),
    tapeValue: t.value(y),
    tapeDeriv: t.adjoint(xv),
    forwardValue: F.value(yf),
    forwardDeriv: F.tangent(yf),
  };
}

function binaryEverywhere(name: BinaryName, a: number, b: number) {
  const p = primitives[name];
  const tableValue = p.value(a, b);

  const t = new Tape();
  const av = t.input(a);
  const bv = t.input(b);
  const y = t[name](av, bv);
  t.backward(y);

  const F1 = new Forward();
  const ya = F1[name](F1.input(a, 1), F1.input(b, 0));
  const F2 = new Forward();
  const yb = F2[name](F2.input(a, 0), F2.input(b, 1));

  return {
    tableValue,
    tableDa: p.da(a, b, tableValue),
    tableDb: p.db(a, b, tableValue),
    plainValue: plain[name](a, b),
    tapeValue: t.value(y),
    tapeDa: t.adjoint(av),
    tapeDb: t.adjoint(bv),
    forwardValue: F1.value(ya),
    forwardDa: F1.tangent(ya),
    forwardDb: F2.tangent(yb),
  };
}

describe('the primitive table', () => {
  test('names every primitive once, split into unary and binary', () => {
    expect([...unaryNames, ...binaryNames].sort()).toEqual(Object.keys(primitives).sort());
    expect(Object.keys(unary).sort()).toEqual([...unaryNames].sort());
    expect(Object.keys(binary).sort()).toEqual([...binaryNames].sort());
    for (const [key, p] of Object.entries(primitives)) expect(p.name).toBe(key);
    for (const name of unaryNames) expect(primitives[name]).toHaveProperty('deriv');
    for (const name of binaryNames) expect(primitives[name]).toHaveProperty('da');
  });

  test('is frozen, so no caller can change a derivative behind every implementation', () => {
    expect(Object.isFrozen(primitives)).toBe(true);
    for (const p of Object.values(primitives)) expect(Object.isFrozen(p)).toBe(true);
  });
});

describe.each(unaryNames)('%s', (name) => {
  const c = unary[name];
  test.each(c.at)('value and derivative at %d', (x) => {
    const r = unaryEverywhere(name, x);
    expectClose(r.tableValue, c.f(x), 1e-12);
    expectClose(r.tableDeriv, c.df(x), 1e-12);
    // Every implementation reads the table: values are identical, not merely close.
    expect(r.plainValue).toBe(r.tableValue);
    expect(r.tapeValue).toBe(r.tableValue);
    expect(r.forwardValue).toBe(r.tableValue);
    expect(r.tapeDeriv).toBe(r.tableDeriv);
    expect(r.forwardDeriv).toBe(r.tableDeriv);
  });
});

describe.each(binaryNames)('%s', (name) => {
  const c = binary[name];
  test.each(c.at)('value and partials at (%d, %d)', (a, b) => {
    const r = binaryEverywhere(name, a, b);
    expectClose(r.tableValue, c.f(a, b), 1e-12);
    expectClose(r.tableDa, c.da(a, b), 1e-12);
    expectClose(r.tableDb, c.db(a, b), 1e-12);
    expect(r.plainValue).toBe(r.tableValue);
    expect(r.tapeValue).toBe(r.tableValue);
    expect(r.forwardValue).toBe(r.tableValue);
    expect(r.tapeDa).toBe(r.tableDa);
    expect(r.tapeDb).toBe(r.tableDb);
    expect(r.forwardDa).toBe(r.tableDa);
    expect(r.forwardDb).toBe(r.tableDb);
  });
});

describe('subgradient conventions and edge cases', () => {
  test('abs at 0 has derivative 0, and NaN in gives NaN out', () => {
    expect(unaryEverywhere('abs', 0).tapeDeriv).toBe(0);
    expect(unaryEverywhere('abs', -0).tapeDeriv).toBe(0);
    expect(unaryEverywhere('abs', NaN).tableDeriv).toBeNaN();
  });

  test('max and min give the whole derivative to the first argument on a tie', () => {
    for (const name of ['max', 'min'] as const) {
      const r = binaryEverywhere(name, 2, 2);
      expect([r.tableValue, r.tapeDa, r.tapeDb, r.forwardDa, r.forwardDb]).toEqual([2, 1, 0, 1, 0]);
    }
  });

  test('max and min propagate NaN in their value', () => {
    expect(primitives.max.value(NaN, 1)).toBeNaN();
    expect(primitives.max.value(1, NaN)).toBeNaN();
    expect(primitives.min.value(NaN, 1)).toBeNaN();
    expect(primitives.min.value(1, NaN)).toBeNaN();
  });

  test('pow: d/db is 0 for a non-positive base', () => {
    expect(binaryEverywhere('pow', 0, 2).tapeDb).toBe(0);
    expect(binaryEverywhere('pow', -2, 3).tapeDb).toBe(0);
    // ...while d/da is the ordinary power rule, also for a negative base and integer exponent.
    expect(binaryEverywhere('pow', -2, 3).tapeDa).toBe(12);
    expect(binaryEverywhere('pow', 0, 2).tapeDa).toBe(0);
  });

  test('pow: d/da is 0 for a zero exponent, including 0^0', () => {
    const r = binaryEverywhere('pow', 0, 0);
    expect([r.tableValue, r.tapeDa, r.tapeDb]).toEqual([1, 0, 0]);
    expect(binaryEverywhere('pow', 5, 0).tapeDa).toBe(0);
    expectClose(binaryEverywhere('pow', 5, 0).tapeDb, Math.log(5));
  });

  test('sqrt at 0 has derivative +Infinity', () => {
    expect(unaryEverywhere('sqrt', 0).tapeDeriv).toBe(Infinity);
  });

  test('sigmoid is stable and accurate in both tails', () => {
    // Below about -709, exp(-x) overflows; the stable form still gives the subnormal answer.
    expect(primitives.sigmoid.value(-720)).toBeGreaterThan(0);
    expectClose(primitives.sigmoid.value(-720), Math.exp(-720), 1e-6);
    expect(primitives.sigmoid.value(800)).toBe(1);
    expect(primitives.sigmoid.value(-800)).toBe(0);
    expect(primitives.sigmoid.value(0)).toBe(0.5);
    // Far out, the derivative is e^-|x| to full relative precision, not 0 or NaN.
    expectClose(primitives.sigmoid.deriv(40, primitives.sigmoid.value(40)), Math.exp(-40), 1e-12);
    expectClose(primitives.sigmoid.deriv(-40, primitives.sigmoid.value(-40)), Math.exp(-40), 1e-12);
    expectClose(primitives.sigmoid.value(-40), Math.exp(-40), 1e-12);
    expect(primitives.sigmoid.deriv(800, 1)).toBe(0);
    expect(primitives.sigmoid.deriv(0, 0.5)).toBe(0.25);
  });

  test('softplus is stable in both tails', () => {
    expect(primitives.softplus.value(800)).toBe(800);
    expectClose(primitives.softplus.value(-40), Math.exp(-40), 1e-12);
    expect(primitives.softplus.value(0)).toBe(Math.LN2);
    expect(primitives.softplus.deriv(800, 800)).toBe(1);
    expectClose(primitives.softplus.deriv(-40, 0), Math.exp(-40), 1e-12);
    expect(primitives.softplus.deriv(0, Math.LN2)).toBe(0.5);
  });

  test('expm1 and log1p keep full precision near 0', () => {
    expectClose(primitives.expm1.value(1e-10), 1e-10 + 5e-21, 1e-15);
    expectClose(primitives.log1p.value(1e-10), 1e-10 - 5e-21, 1e-15);
    expect(primitives.expm1.deriv(0, 0)).toBe(1);
    expect(primitives.log1p.deriv(0, 0)).toBe(1);
  });

  test('log and recip at 0', () => {
    expect(unaryEverywhere('log', 0)).toMatchObject({ tableValue: -Infinity, tapeDeriv: Infinity });
    expect(unaryEverywhere('recip', 0)).toMatchObject({ tableValue: Infinity });
  });
});
