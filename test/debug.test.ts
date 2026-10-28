/*
 * The opt-in debug mode of Tape and Forward: every operand is checked to be a live variable of
 * that tape (or Forward), and a mistake throws at once instead of reading the wrong node.
 */

import { describe, expect, test } from 'vitest';
import {
  type Arithmetic,
  type Dual,
  Forward,
  gradient,
  jvp,
  logsumexp,
  type Mark,
  softmax,
  Tape,
  type Var,
} from '../src/index.ts';

const f = <T>(A: Arithmetic<T>, [x, y]: readonly [T, T]): T =>
  A.add(logsumexp(A, [A.mulc(x, 2), A.csub(1, y), A.sin(x)]), A.powc(A.addc(y, 3), 1.5));

describe('Tape in debug mode', () => {
  test('computes exactly what an ordinary tape does', () => {
    const plainTape = gradient(f, [0.3, -0.7]);
    const debugTape = gradient(f, [0.3, -0.7], new Tape({ debug: true }));
    expect(debugTape).toEqual(plainTape);
  });

  test('every method works on its own handles', () => {
    const t = new Tape({ debug: true, capacity: 2 }); // and keeps working as the tape grows
    const [x, y] = t.inputs([2, 3]);
    const c = t.constant(4);
    const z = t.add(t.mul(x, y), t.divc(c, 2));
    expect(t.value(z)).toBe(8);
    t.backward(z);
    expect([t.adjoint(x), t.adjoint(y), t.adjoint(c)]).toEqual([3, 2, 0.5]);
    expect(Array.from(t.gradient(z, [x, y]))).toEqual([3, 2]);
    t.vjp([z, x], [1, 1]);
    expect(Array.from(t.adjoints([x, y]))).toEqual([4, 2]);
    expect(t.value(t.input(5))).toBe(5);
    const weights = softmax(t, [x, y]).map((s) => t.value(s));
    expect(weights[0]! + weights[1]!).toBeCloseTo(1, 15);
    expect(t.value(t.cdiv(1, x)) + t.value(t.subc(x, 1)) + t.value(t.powc(x, 2))).toBe(0.5 + 1 + 4);
    expect(t.size).toBeGreaterThan(2);
  });

  test('keeps track of slots well past its first allocation', () => {
    const t = new Tape({ debug: true });
    const x = t.input(1.001);
    let y = x;
    for (let i = 0; i < 1000; i++) y = t.mul(y, x);
    t.backward(y);
    expect(t.adjoint(x)).toBeCloseTo(1001 * 1.001 ** 1000, 9);
  });

  test('rejects a plain number, naming the method and the argument', () => {
    const t = new Tape({ debug: true });
    t.input(1);
    expect(() => t.exp(1 as Var)).toThrow(
      'Tape.exp: argument 1 is the plain number 1, not a variable of this Tape',
    );
    expect(() => t.add(t.input(2), 3 as Var)).toThrow(
      /Tape\.add: argument 2 is the plain number 3/,
    );
    expect(() => t.value(0 as Var)).toThrow(/plain number 0/);
  });

  test('rejects anything that is not a handle at all', () => {
    const t = new Tape({ debug: true });
    for (const bad of [NaN, -1, 1.5, Infinity, 2 ** 60]) {
      expect(() => t.exp(bad as Var), String(bad)).toThrow(/is not a variable/);
    }
  });

  test('rejects a variable from before clear()', () => {
    const t = new Tape({ debug: true });
    const x = t.input(1);
    t.clear();
    expect(() => t.exp(x)).toThrow(
      /Tape\.exp: argument 1 is stale: its node was discarded by clear\(\) or rewind\(\)/,
    );
    t.input(2); // the slot is in use again
    expect(() => t.exp(x)).toThrow(/stale/);
  });

  test('keeps variables from before a mark valid across rewind(), and rejects later ones', () => {
    const t = new Tape({ debug: true });
    const x = t.input(2);
    const m: Mark = t.mark();
    const y = t.square(x);
    t.rewind(m);
    expect(t.value(t.exp(x))).toBe(Math.exp(2));
    expect(() => t.value(y)).toThrow(/stale/);
    expect(() => {
      t.backward(y);
    }).toThrow(/Tape\.backward: argument 1 is stale/);
  });

  test('rejects a variable of another tape, or of a Forward', () => {
    const t = new Tape({ debug: true });
    const other = new Tape({ debug: true });
    const x = other.input(1);
    t.input(1);
    expect(() => t.exp(x)).toThrow(
      /Tape\.exp: argument 1 is a variable of another Tape or Forward/,
    );
    const F = new Forward({ debug: true });
    const d = F.input(1, 1);
    expect(() => t.exp(d as unknown as Var)).toThrow(/another Tape or Forward/);
  });

  test('checks every argument of the sweep and read methods', () => {
    const t = new Tape({ debug: true });
    const x = t.input(1);
    const bad = 7 as Var;
    expect(() => {
      t.vjp([x, bad], [1, 1]);
    }).toThrow(/Tape\.vjp: argument 1, element 2 is the plain number 7/);
    expect(() => t.adjoints([bad])).toThrow(/Tape\.adjoints: argument 1, element 1/);
    expect(() => t.gradient(x, [bad])).toThrow(/Tape\.gradient: argument 2, element 1/);
    expect(() => t.gradient(bad, [x])).toThrow(/Tape\.gradient: argument 1 is the plain number 7/);
    expect(() => t.adjoint(bad)).toThrow(/Tape\.adjoint/);
    expect(() => t.csub(1, bad)).toThrow(/Tape\.csub: argument 2/);
    expect(() => t.mulc(bad, 1)).toThrow(/Tape\.mulc: argument 1/);
  });

  test('rejects a variable where a plain number belongs: the typeof trap', () => {
    const t = new Tape({ debug: true });
    const x = t.input(3);
    // A test such as typeof v === 'number' sends a variable to constant(); debug mode notices.
    const lift = (v: Var | number): Var => (typeof v === 'number' ? t.constant(v) : v);
    expect(() => lift(x)).toThrow(
      /Tape\.constant: argument 1 is a variable, where a plain number belongs/,
    );
    expect(() => t.mulc(x, x)).toThrow(/Tape\.mulc: argument 2 is a variable/);
    expect(() => t.csub(x, x)).toThrow(/Tape\.csub: argument 1 is a variable/);
    expect(() => t.input(x)).toThrow(/Tape\.input: argument 1 is a variable/);
    expect(() => t.inputs([1, x])).toThrow(/Tape\.inputs: argument 1, element 2 is a variable/);
    expect(() => t.inputs(Float64Array.from([x]))).toThrow(/element 1 is a variable/);
    expect(() => {
      t.backward(x, x);
    }).toThrow(/Tape\.backward: argument 2 is a variable/);
    const F = new Forward({ debug: true });
    const d = F.input(1, 0);
    expect(() => F.input(1, d)).toThrow(/Forward\.input: argument 2 is a variable/);
    // Another instance's variable is not recognised as one: no ordinary number may be refused.
    expect(F.value(F.constant(x))).toBe(x);
    // Ordinary numbers, large ones included, are fine.
    expect(t.value(t.constant(2 ** 31))).toBe(2 ** 31);
    expect(t.value(t.constant(2 ** 40 + 0.5))).toBe(2 ** 40 + 0.5);
    expect(t.value(t.mulc(x, -(2 ** 31)))).toBe(-3 * 2 ** 31);
    expect(t.value(t.constant(2 ** 53))).toBe(2 ** 53);
    expect(t.value(t.constant(x + 1000))).toBe(x + 1000); // no slot written there
  });

  test("passes arguments beyond a method's own through unchecked", () => {
    const t = new Tape({ debug: true });
    const x = t.input(3);
    const value = t.value.bind(t) as (...args: unknown[]) => number;
    expect(value(x, 'ignored')).toBe(3);
  });

  test('is off by default: a plain number then silently reads whatever node has that index', () => {
    const t = new Tape();
    t.input(5);
    expect(t.value(1 as Var)).toBe(5);
  });
});

describe('Forward in debug mode', () => {
  test('computes exactly what an ordinary Forward does', () => {
    expect(jvp(f, [0.3, -0.7], [1, 2], new Forward({ debug: true }))).toEqual(
      jvp(f, [0.3, -0.7], [1, 2]),
    );
  });

  test('rejects plain numbers, stale values and values of another Forward', () => {
    const F = new Forward({ debug: true });
    const x = F.input(1, 1);
    expect(F.tangent(F.exp(x))).toBe(Math.E);
    expect(F.value(F.constant(2))).toBe(2);
    expect(() => F.exp(0 as Dual)).toThrow(
      'Forward.exp: argument 1 is the plain number 0, not a variable of this Forward',
    );
    expect(() => F.tangent(5 as Dual)).toThrow(/Forward\.tangent/);
    const G = new Forward({ debug: true });
    expect(() => G.exp(x)).toThrow(/another Tape or Forward/);
    F.clear();
    expect(() => F.value(x)).toThrow(/stale/);
    F.input(2, 0);
    expect(() => F.mul(x, x)).toThrow(/Forward\.mul: argument 1 is stale/);
  });
});
