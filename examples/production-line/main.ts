/*
 * The production-line example, end to end: `runExample` is the walkthrough, `run.ts` runs it with
 * the default line (`npm run example`), and the tests run it on a small one.
 * See README.md beside this file for what it shows and why.
 */

import { defaultKnobs, type Line } from './model.ts';
import { apply, describe, flags, type Schedule } from './schedule.ts';
import {
  localSearch,
  measurePassCost,
  Problem,
  rankMoves,
  type SearchResult,
  totalWork,
  workToReach,
} from './search.ts';

export interface ExampleOptions {
  readonly line: Line;
  readonly start: Schedule;
  readonly maxShift: number;
  readonly maxTransfer: number;
  /** Seeds of the random-order searches to compare with. */
  readonly seeds: readonly number[];
  /**
   * What a gradient pass costs in plain evaluations. Measured by timing when not given; the tests
   * fix it so that their output does not depend on the machine.
   */
  readonly passCost?: number;
  readonly log: (line: string) => void;
}

/** Two 20-minute windows per station, placed without much thought: the schedule to improve. */
export const initialSchedule: Schedule = [
  { station: 0, start: 40, length: 20 },
  { station: 0, start: 300, length: 20 },
  { station: 1, start: 60, length: 20 },
  { station: 1, start: 360, length: 20 },
  { station: 2, start: 200, length: 20 },
  { station: 2, start: 420, length: 20 },
];

export function runExample(options: ExampleOptions): {
  gradientOrdered: SearchResult;
  randomOrdered: SearchResult[];
  /** d(profit)/d(price) at the optimised schedule. */
  price: number;
} {
  const { line, start, log } = options;
  const fmt = (x: number, digits = 2): string => x.toFixed(digits);
  const problem = new Problem(line, defaultKnobs);

  log(`A line of ${String(line.stations)} stations over a ${String(line.minutes)}-minute shift;`);
  log(
    `${String(start.length)} maintenance windows, ${String(start.reduce((s, w) => s + w.length, 0))} minutes in all.\n`,
  );

  // 1. Plain evaluation.
  log(`1. Profit of the starting schedule (plain evaluation): ${fmt(problem.profit(start))}`);

  // 2. One forward and one backward pass: the gradient with respect to every minute's pause flag.
  const { value, gradient } = problem.gradient(flags(line, start));
  log(`2. One tape pass: profit ${fmt(value)} and ${String(gradient.length)} partial derivatives.`);
  for (let s = 0; s < line.stations; s++) {
    const row = gradient.subarray(s * line.minutes, (s + 1) * line.minutes);
    let best = 0;
    for (let t = 1; t < row.length; t++) if (row[t]! > row[best]!) best = t;
    log(
      `   station ${String(s)}: d(profit)/d(pause) ranges ${fmt(Math.min(...row))} .. ${fmt(Math.max(...row))}` +
        `, least harmful minute to stop: ${String(best)}`,
    );
  }

  // 3. First-order estimates of every candidate move, ranked.
  const ranked = rankMoves(problem, start, gradient, options.maxShift, options.maxTransfer);
  const base = problem.profit(start);
  const actual = ranked.map((r) => problem.profit(apply(start, r.move)) - base);
  log(
    `\n3. ${String(ranked.length)} candidate moves, ranked by their first-order estimate. The top five:`,
  );
  ranked.slice(0, 5).forEach((r, i) => {
    log(
      `   ${describe(r.move).padEnd(37)} estimated ${fmt(r.estimate).padStart(7)}   actual ${fmt(actual[i]!).padStart(7)}`,
    );
  });
  const top = Math.min(20, ranked.length);
  const improving = (xs: readonly number[]): number => xs.filter((x) => x > 1e-9).length;
  log(
    `   ${String(improving(actual.slice(0, top)))} of the top ${String(top)} improve profit; ` +
      `${String(improving(actual))} of all ${String(ranked.length)} do.`,
  );

  // 4. First-improvement local search, in gradient order and in random order. Work is counted in
  // evaluation equivalents: full evaluations, plus each gradient pass at what one costs measured
  // against a plain evaluation on this machine.
  const passCost = options.passCost ?? measurePassCost(line, defaultKnobs, start);
  const gradientOrdered = localSearch(problem, start, { ...options, order: 'gradient', seed: 0 });
  const randomOrdered = options.seeds.map((seed) =>
    localSearch(problem, start, { ...options, order: 'random', seed }),
  );
  const all = [gradientOrdered, ...randomOrdered];
  const best = Math.max(...all.map((r) => r.profit));
  const levels = [0.5, 0.75, 0.85, 0.9];
  const work = (r: SearchResult): (number | undefined)[] => [
    ...levels.map((f) => workToReach(r, base + f * (best - base), passCost)),
    workToReach(r, r.profit, passCost),
    totalWork(r, passCost),
  ];
  const headings = [...levels.map((f) => `${String(f * 100)}%`), 'own opt.', 'in all'];
  const cell = (x: number | undefined): string => (x === undefined ? '-' : fmt(x, 0)).padStart(9);
  log(
    `\n4. First-improvement local search from the same start. Work is in evaluation equivalents:\n` +
      `   full evaluations, plus ${fmt(passCost, 1)} for each gradient pass (its cost here, measured). ` +
      `The best optimum\n   any search found is ${fmt(best)}; the columns give the work to reach a ` +
      "share of its gain, then each\n   search's own optimum, then all the work, the final round included.",
  );
  log(
    `   ${'order'.padEnd(17)} ${'optimum'.padStart(8)} ${'moves'.padStart(6)}` +
      headings.map((h) => h.padStart(9)).join(''),
  );
  all.forEach((r, i) => {
    const name = i === 0 ? 'gradient' : `random (seed ${String(options.seeds[i - 1])})`;
    log(
      `   ${name.padEnd(17)} ${fmt(r.profit).padStart(8)} ${String(r.improvements).padStart(6)}` +
        work(r).map(cell).join(''),
    );
  });
  const median = (xs: readonly (number | undefined)[]): number | undefined => {
    const got = xs.filter((x) => x !== undefined).sort((a, b) => a - b);
    // A search that never got there counts as slower than any that did.
    return xs.length - got.length > xs.length / 2 ? undefined : got[Math.floor(xs.length / 2)];
  };
  const g = work(gradientOrdered);
  const rs = randomOrdered.map(work);
  log('   Gradient order against the median random search:');
  headings.forEach((h, k) => {
    const mine = g[k];
    const theirs = median(rs.map((w) => w[k]));
    const verdict =
      mine === undefined
        ? 'never got there'
        : theirs === undefined || mine < theirs
          ? `faster (${fmt(mine, 0)} against ${theirs === undefined ? 'never' : fmt(theirs, 0)})`
          : `slower (${fmt(mine, 0)} against ${fmt(theirs, 0)})`;
    log(`     ${h.padEnd(9)} ${verdict}`);
  });
  log('   optimised windows (gradient order):');
  for (const w of gradientOrdered.schedule) {
    log(
      `     station ${String(w.station)}: minutes ${String(w.start)}-${String(w.start + w.length - 1)} (${String(w.length)} min)`,
    );
  }

  // 5. Forward mode: sensitivity of profit to one parameter.
  const price = problem.sensitivity(gradientOrdered.schedule, 'price');
  const wear = problem.sensitivity(gradientOrdered.schedule, 'wearRate');
  log('\n5. Forward mode, one pass each, at the optimised schedule:');
  log(
    `   d(profit)/d(price)    = ${fmt(price.derivative)}  (good parts sold, weighted by what each fetched)`,
  );
  log(
    `   d(profit)/d(wearRate) = ${fmt(wear.derivative, 0)}  (so wearing 0.001 faster per minute costs ` +
      `${fmt(-wear.derivative / 1000)})`,
  );

  return { gradientOrdered, randomOrdered, price: price.derivative };
}
