/*
 * The production-line example (examples/production-line): a smoke test of the whole walkthrough on
 * a small line, and its gradient checked against finite differences.
 */

import { expect, test } from 'vitest';
import { checkGradient, evaluate, plain, type ScalarFunction } from '../src/index.ts';
import { runExample } from '../examples/production-line/main.ts';
import {
  defaultKnobs,
  defaultLine,
  type Line,
  profile,
  profit,
} from '../examples/production-line/model.ts';
import { flags, type Schedule } from '../examples/production-line/schedule.ts';
import {
  measurePassCost,
  Problem,
  rng,
  totalWork,
  workToReach,
} from '../examples/production-line/search.ts';

const small: Line = {
  ...defaultLine,
  stations: 2,
  minutes: 60,
  capacity: [1.1, 1],
  initialWear: [0.5, 0.9],
  supply: profile(60, 1.05, 0.1, [[20, 30]]),
  value: profile(60, 1, 0.4, [[40, 50]]),
};
const smallSchedule: Schedule = [
  { station: 0, start: 5, length: 8 },
  { station: 1, start: 30, length: 6 },
];

test('the walkthrough runs end to end and improves the schedule', () => {
  const lines: string[] = [];
  const r = runExample({
    line: small,
    start: smallSchedule,
    maxShift: 3,
    maxTransfer: 2,
    seeds: [1, 2],
    passCost: 2.5,
    log: (l) => lines.push(l),
  });
  const start = new Problem(small, defaultKnobs).profit(smallSchedule);
  expect(r.gradientOrdered.profit).toBeGreaterThan(start);
  for (const x of r.randomOrdered) expect(x.profit).toBeGreaterThan(start);
  expect(r.gradientOrdered.gradientPasses).toBe(r.gradientOrdered.improvements + 1);
  // Work counts every gradient pass at its cost, up to the point measured.
  const g = r.gradientOrdered;
  expect(totalWork(g, 2.5)).toBe(g.evaluations + 2.5 * g.gradientPasses);
  const last = g.trace.at(-1)!;
  expect(last.gradientPasses).toBe(g.improvements);
  expect(workToReach(g, g.profit, 2.5)).toBe(last.evaluations + 2.5 * last.gradientPasses);
  expect(workToReach(g, Infinity, 2.5)).toBeUndefined();
  expect(lines.join('\n')).toMatch(/plus 2\.5 for each gradient pass/);
  expect(lines.join('\n')).toMatch(/^ {5}in all {4}(faster|slower) \(/m);
  expect(lines.join('\n')).toMatch(/^5\. Forward mode/m);
  // Profit is linear in the price, so its derivative is the difference a unit of price makes.
  const at = (price: number): number =>
    profit(plain, small, flags(small, r.gradientOrdered.schedule), {
      ...defaultKnobs,
      price,
    });
  expect(r.price).toBeCloseTo(at(defaultKnobs.price + 1) - at(defaultKnobs.price), 9);
});

test('the gradient of profit in the relaxed pause flags and the knobs matches finite differences', () => {
  // Pause flags anywhere in [0, 1], then the price and the wear rate.
  const random = rng(7);
  const pause = Array.from({ length: small.stations * small.minutes }, () => random());
  const f: ScalarFunction = (A, xs) =>
    profit(A, small, xs.slice(0, pause.length), {
      price: xs[pause.length]!,
      wearRate: xs[pause.length + 1]!,
    });
  const at = [...pause, defaultKnobs.price, defaultKnobs.wearRate];
  const r = checkGradient(f, at, { rtol: 1e-5, atol: 1e-6 });
  expect(r.ok).toBe(true);
  expect(evaluate(f, at)).toBe(r.value);
});

test('Problem.gradient is the same gradient, in the flags only', () => {
  const p = new Problem(small, defaultKnobs);
  const pause = flags(small, smallSchedule);
  const { value, gradient } = p.gradient(pause);
  const f: ScalarFunction = (A, xs) =>
    profit(A, small, xs, {
      price: A.constant(defaultKnobs.price),
      wearRate: A.constant(defaultKnobs.wearRate),
    });
  const r = checkGradient(f, pause);
  expect(value).toBe(r.value);
  expect(Array.from(gradient)).toEqual(r.analytic);
  expect(r.ok).toBe(true);
});

test('the measured cost of a gradient pass is a plausible multiple of an evaluation', () => {
  const cost = measurePassCost(small, defaultKnobs, smallSchedule, 5);
  expect(cost).toBeGreaterThan(0.5);
  expect(cost).toBeLessThan(50);
});
