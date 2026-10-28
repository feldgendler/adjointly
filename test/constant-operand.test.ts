/*
 * The operations with a plain-number operand (addc, subc, csub, mulc, divc, cdiv, powc): the same
 * values and derivatives as spelling the number with constant(), in every arithmetic, bit for bit,
 * for one tape node instead of two.
 */

import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import { constantOperandOps, type ConstantOperandName } from '../src/arithmetic.ts';
import {
  type Arithmetic,
  checkGradient,
  Forward,
  gradient,
  plain,
  type ScalarFunction,
  Tape,
} from '../src/index.ts';

const names = Object.keys(constantOperandOps) as ConstantOperandName[];

/** `op` with the number `c`, written with the new operation. */
function withNumber<T>(A: Arithmetic<T>, op: ConstantOperandName, x: T, c: number): T {
  return constantOperandOps[op].constantFirst
    ? A[op](c as never, x as never)
    : A[op](x as never, c as never);
}

/** The same, written as the binary primitive with a constant node. */
function withConstant<T>(A: Arithmetic<T>, op: ConstantOperandName, x: T, c: number): T {
  const { of, constantFirst } = constantOperandOps[op];
  return constantFirst ? A[of](A.constant(c), x) : A[of](x, A.constant(c));
}

/** Value and derivative with respect to x, by every route. */
function everyway(op: ConstantOperandName, x: number, c: number) {
  const tapes = [new Tape(), new Tape()];
  const rev = [withNumber, withConstant].map((spell, k) => {
    const t = tapes[k]!;
    const xv = t.input(x);
    const before = t.size;
    const y = spell(t, op, xv, c);
    const nodes = t.size - before;
    t.backward(y);
    return { value: t.value(y), d: t.adjoint(xv), nodes };
  });
  const fwd = [withNumber, withConstant].map((spell) => {
    const F = new Forward();
    const y = spell(F, op, F.input(x, 1), c);
    return { value: F.value(y), d: F.tangent(y) };
  });
  return {
    plain: [withNumber(plain, op, x, c), withConstant(plain, op, x, c)],
    rev,
    fwd,
  };
}

/** Equal as numbers, NaN equal to NaN. */
const same = (a: number, b: number): boolean => a === b || (Number.isNaN(a) && Number.isNaN(b));

const specials = [0, -0, 0.5, -1.5, 2, 3, 1e-300, -1e300, Infinity, -Infinity, NaN];

describe.each(names)('%s', (op) => {
  test('matches the constant-node spelling in value and derivative, in every arithmetic', () => {
    for (const x of specials) {
      for (const c of specials) {
        const r = everyway(op, x, c);
        const where = `${op} at x = ${String(x)}, c = ${String(c)}`;
        const v = r.plain[1]!;
        // Values are bit-identical: the same table function computes them.
        for (const got of [r.plain[0]!, r.rev[0]!.value, r.rev[1]!.value, r.fwd[0]!.value]) {
          expect(Object.is(got, v), where).toBe(true);
        }
        // Derivatives are the same partial from the same table entry.
        expect(same(r.rev[0]!.d, r.rev[1]!.d), `${where}: reverse`).toBe(true);
        expect(same(r.fwd[0]!.d, r.fwd[1]!.d), `${where}: forward`).toBe(true);
        expect(same(r.rev[0]!.d, r.fwd[0]!.d), `${where}: reverse vs forward`).toBe(true);
      }
    }
  });

  test('records one node, where the constant spelling records two', () => {
    const r = everyway(op, 1.5, 2);
    expect(r.rev[0]!.nodes).toBe(1);
    expect(r.rev[1]!.nodes).toBe(2);
  });
});

test('the operations compute what their names say', () => {
  const [x, c] = [5, 2];
  expect([
    plain.addc(x, c),
    plain.subc(x, c),
    plain.csub(c, x),
    plain.mulc(x, c),
    plain.divc(x, c),
    plain.cdiv(c, x),
    plain.powc(x, c),
  ]).toEqual([7, 3, -3, 10, 2.5, 0.4, 25]);
  expect(gradient((A, [y]) => A.csub(1, A.cdiv(2, y)), [4]).gradient).toEqual([2 / 16]);
});

test('the zero rule holds through a zero constant', () => {
  // mulc(sqrt(x), 0) at 0: the partial 0 annihilates sqrt's infinite derivative in both modes.
  const f: ScalarFunction<[number]> = (A, [x]) => A.mulc(A.sqrt(x), 0);
  expect(gradient(f, [0]).gradient).toEqual([0]);
  const F = new Forward();
  expect(F.tangent(f(F, [F.input(0, 1)]))).toBe(0);
});

test('random expressions agree with the constant spelling and with finite differences', () => {
  const step = fc.record({
    op: fc.constantFrom(...names),
    c: fc.double({ min: -2, max: 2, noNaN: true }),
  });
  fc.assert(
    fc.property(
      fc.array(step, { minLength: 1, maxLength: 12 }),
      fc.double({ min: 0.5, max: 2, noNaN: true }),
      (steps, x0) => {
        // Keep each step inside its domain and away from overflow: a positive operand for a power
        // or a division into a constant, a nonzero divisor, and a bounded result.
        const run =
          (spell: typeof withNumber) =>
          <T>(A: Arithmetic<T>, [x]: readonly [T]): T => {
            let y = x;
            for (const { op, c } of steps) {
              const safe =
                op === 'powc' || op === 'cdiv' ? A.addc(A.square(A.tanh(y)), 0.5) : A.tanh(y);
              const k = op === 'divc' ? (Math.abs(c) < 0.25 ? 0.25 : c) : c;
              y = spell(A, op, safe, k);
            }
            return y;
          };
        const a = gradient(run(withNumber), [x0]);
        const b = gradient(run(withConstant), [x0]);
        expect(Object.is(a.value, b.value)).toBe(true);
        expect(same(a.gradient[0]!, b.gradient[0]!)).toBe(true);
        const check = checkGradient(run(withNumber), [x0], { rtol: 1e-5, atol: 1e-7 });
        expect(check.ok, JSON.stringify(check.worst)).toBe(true);
      },
    ),
    { numRuns: 300 },
  );
});
