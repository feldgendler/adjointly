/* The tape's lifecycle: growth, clear, mark/rewind, backward sweeps, VJPs and adjoint reads. */

import { describe, expect, test } from 'vitest';
import { type Mark, Tape, type Var } from '../src/index.ts';
import { expectClose } from './support/util.ts';

describe('construction and growth', () => {
  test('starts empty with the requested capacity', () => {
    expect(new Tape().capacity).toBe(1024);
    expect(new Tape({ capacity: 3 }).capacity).toBe(3);
    expect(new Tape({ capacity: 1 }).capacity).toBe(1);
    for (const capacity of [0, -1, 2.7, NaN, Infinity]) {
      expect(() => new Tape({ capacity }), String(capacity)).toThrow(RangeError);
    }
    expect(new Tape().size).toBe(0);
  });

  test('grows past its initial capacity by doubling, keeping every node', () => {
    const t = new Tape({ capacity: 1 });
    const x = t.input(1.0001);
    let y: Var = x;
    for (let i = 0; i < 1000; i++) y = t.mul(y, x); // x^1001
    expect(t.size).toBe(1001);
    expect(t.capacity).toBe(1023); // slots double from 2 (one node plus the sink) to 1024
    expectClose(t.value(y), Math.pow(1.0001, 1001), 1e-12);
    t.backward(y);
    expectClose(t.adjoint(x), 1001 * Math.pow(1.0001, 1000), 1e-12);
  });

  test('adjoints from a sweep survive recording that grows the tape', () => {
    const t = new Tape({ capacity: 4 });
    const x = t.input(3);
    t.backward(t.square(x));
    expect(t.adjoint(x)).toBe(6);
    for (let i = 0; i < 10; i++) t.constant(i); // grows the tape twice
    expect(t.capacity).toBeGreaterThan(4);
    expect(t.adjoint(x)).toBe(6);
  });

  test('a backward sweep after growth sees the old nodes and the new ones', () => {
    const t = new Tape({ capacity: 2 });
    const a = t.input(3);
    const b = t.input(4);
    t.backward(t.mul(a, b)); // sweeps on a tape that has just grown
    expect(t.adjoints([a, b])).toEqual(Float64Array.from([4, 3]));
  });
});

describe('inputs, constants and values', () => {
  test('input, inputs and constant record leaves with their values', () => {
    const t = new Tape();
    const x = t.input(2);
    const ys = t.inputs([3, 4]);
    const c = t.constant(5);
    expect([x, ...ys, c].map((v) => t.value(v))).toEqual([2, 3, 4, 5]);
    expect(t.size).toBe(4);
    expect(t.inputs(Float64Array.from([6]))).toHaveLength(1);
  });

  test('a constant accumulates an adjoint like any leaf', () => {
    const t = new Tape();
    const x = t.input(2);
    const c = t.constant(5);
    t.backward(t.mul(x, c));
    expect([t.adjoint(x), t.adjoint(c)]).toEqual([5, 2]);
  });
});

describe('backward', () => {
  test('the seed scales every adjoint', () => {
    const t = new Tape();
    const x = t.input(3);
    const y = t.square(x);
    t.backward(y, 0.5);
    expect(t.adjoint(x)).toBe(3);
    expect(t.adjoint(y)).toBe(0.5);
  });

  test('a variable used several times accumulates every path', () => {
    const t = new Tape();
    const x = t.input(2);
    // y = x * x + x = x^2 + x, dy/dx = 2x + 1
    t.backward(t.add(t.mul(x, x), x));
    expect(t.adjoint(x)).toBe(5);
  });

  test('several sweeps over one tape each start from zero adjoints', () => {
    const t = new Tape();
    const x = t.input(2);
    const y = t.input(3);
    const f = t.mul(x, y);
    const g = t.add(x, t.square(y));
    t.backward(f);
    expect(t.adjoints([x, y])).toEqual(Float64Array.from([3, 2]));
    t.backward(g);
    expect(t.adjoints([x, y])).toEqual(Float64Array.from([1, 6]));
    t.backward(f);
    expect(t.adjoints([x, y])).toEqual(Float64Array.from([3, 2]));
  });

  test('nodes the output does not depend on have adjoint 0', () => {
    const t = new Tape();
    const x = t.input(2);
    const unused = t.input(7);
    const y = t.exp(x);
    t.backward(y);
    expect(t.adjoint(unused)).toBe(0);
  });

  test('a node with zero adjoint passes nothing on, even through an infinite partial', () => {
    const t = new Tape();
    const x = t.input(0);
    const y = t.input(3);
    const s = t.sqrt(x); // partial +Infinity
    const f = t.add(t.mul(s, t.constant(0)), y); // s contributes with weight exactly 0
    t.backward(f);
    expect(t.adjoint(s)).toBe(0);
    expect(t.adjoint(x)).toBe(0); // not 0 * Infinity = NaN
    expect(t.adjoint(y)).toBe(1);
  });

  test('a nonzero adjoint through an infinite partial gives an infinite gradient', () => {
    const t = new Tape();
    const x = t.input(0);
    t.backward(t.sqrt(x));
    expect(t.adjoint(x)).toBe(Infinity);
  });

  test('rejects an output that is not on the tape', () => {
    const t = new Tape();
    t.input(1);
    expect(() => {
      t.backward((t.size + 1) as Var); // one past the last node
    }).toThrow(RangeError);
    expect(() => {
      t.backward(0 as Var); // the sink slot is not a node
    }).toThrow(RangeError);
    expect(() => {
      t.backward(-1 as Var);
    }).toThrow(/not a node/);
  });

  test('gradient is backward plus adjoints, into a given buffer', () => {
    const t = new Tape();
    const [x, y] = t.inputs([2, 5]);
    const out = new Float64Array(2);
    const r = t.gradient(t.div(x, y), [y, x], out);
    expect(r).toBe(out);
    expect(Array.from(out)).toEqual([-2 / 25, 1 / 5]);
    expect(Array.from(t.gradient(t.div(x, y), [x]))).toEqual([1 / 5]);
  });
});

describe('vjp', () => {
  test('is the weighted sum of the outputs’ gradients, in one sweep', () => {
    const t = new Tape();
    const [x, y] = t.inputs([2, 3]);
    const f = t.mul(x, y); // grad (3, 2)
    const g = t.sin(x); // grad (cos 2, 0)
    t.vjp([f, g], [2, -1]);
    expectClose(t.adjoint(x), 2 * 3 - Math.cos(2));
    expect(t.adjoint(y)).toBe(4);
  });

  test('sweeps from the latest output whatever the order given', () => {
    const t = new Tape();
    const x = t.input(2);
    const f = t.square(x);
    const g = t.square(f); // x^4
    t.vjp([g, f], [1, 1]);
    expect(t.adjoint(x)).toBe(4 * 8 + 2 * 2);
  });

  test('an output listed twice gets the sum of its weights', () => {
    const t = new Tape();
    const x = t.input(2);
    const f = t.square(x);
    t.vjp([f, f], [1, 2]);
    expect(t.adjoint(x)).toBe(12);
  });

  test('earlier adjoints are cleared', () => {
    const t = new Tape();
    const x = t.input(2);
    const f = t.square(x);
    t.backward(f);
    t.vjp([x], [1]);
    expect(t.adjoint(x)).toBe(1);
    expect(t.adjoint(f)).toBe(1); // beyond this sweep: left over, documented as meaningless
  });

  test('takes any number of outputs', () => {
    const t = new Tape();
    const x = t.input(2);
    const outs = Array.from({ length: 200_000 }, () => t.square(x));
    t.vjp(outs, new Float64Array(outs.length).fill(1));
    expect(t.adjoint(x)).toBe(4 * 200_000);
  });

  test('with no outputs every adjoint is zero', () => {
    const t = new Tape();
    const x = t.input(2);
    const y = t.square(x);
    t.backward(y);
    t.vjp([], []);
    expect([t.adjoint(x), t.adjoint(y)]).toEqual([0, 0]);
  });

  test('rejects mismatched lengths and foreign outputs', () => {
    const t = new Tape();
    const x = t.input(2);
    expect(() => {
      t.vjp([x], [1, 2]);
    }).toThrow(/1 outputs but 2 weights/);
    expect(() => {
      t.vjp([x, 5 as Var], [1, 1]);
    }).toThrow(/not a node/);
    expect(() => {
      t.vjp([(t.size + 1) as Var], [1]);
    }).toThrow(/not a node/);
    expect(() => {
      t.vjp([-1 as Var], [1]);
    }).toThrow(RangeError);
  });
});

describe('clear, mark and rewind', () => {
  test('clear empties the tape and keeps the storage', () => {
    const t = new Tape({ capacity: 4 });
    for (let i = 0; i < 10; i++) t.input(i);
    const cap = t.capacity;
    t.clear();
    expect(t.size).toBe(0);
    expect(t.capacity).toBe(cap);
    const x = t.input(9);
    expect(t.size).toBe(1);
    expect(t.value(x)).toBe(9);
  });

  test('rewind reuses a prefix across evaluations', () => {
    const t = new Tape();
    const w = t.input(0.5);
    const b = t.input(-1);
    const m = t.mark();
    const grads: number[][] = [];
    for (const x of [1, 2, 3]) {
      t.rewind(m);
      const loss = t.square(t.add(t.mul(w, t.constant(x)), b));
      t.backward(loss);
      grads.push([t.adjoint(w), t.adjoint(b)]);
      expect(t.size).toBe(6);
    }
    // d/dw (wx + b)^2 = 2(wx + b)x, d/db = 2(wx + b)
    expect(grads).toEqual([1, 2, 3].map((x) => [2 * (0.5 * x - 1) * x, 2 * (0.5 * x - 1)]));
    expect(t.value(w)).toBe(0.5);
  });

  test('rewind to the current end is a no-op, and to a mark of the empty tape is clear', () => {
    const t = new Tape();
    const empty = t.mark();
    t.input(1);
    t.rewind(t.mark());
    expect(t.size).toBe(1);
    t.rewind(empty);
    expect(t.size).toBe(0);
  });

  test('rewind refuses a mark beyond the end', () => {
    const t = new Tape();
    t.input(1);
    t.input(2);
    const m = t.mark();
    t.clear();
    expect(() => {
      t.rewind(m);
    }).toThrow(/outside the tape/);
    expect(() => {
      t.rewind(0 as Mark);
    }).toThrow(RangeError);
  });
});
