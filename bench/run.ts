/*
 * Benchmark: each workload evaluated plainly, recorded on a tape, recorded and swept backwards,
 * and in forward mode along one direction. Not part of the tests.
 *
 *   npm run bench            measure and print
 *   npm run bench:compare    ...and compare with bench/baseline.json (CI runs this)
 *   npm run bench:update     ...and overwrite bench/baseline.json with this run
 *
 * What is compared is each mode's time as a multiple of the plain evaluation of the same workload
 * on the same machine in the same run. Absolute times differ between a laptop and a CI runner by
 * more than any regression worth catching; the overhead of differentiation over plain arithmetic
 * is what this library controls, and it moves much less between machines. A ratio more than
 * WARN (default 1.15) times its baseline warns; more than FAIL (default 1.4) times fails.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { Bench } from 'tinybench';
import { Forward, plain, Tape } from '../src/index.ts';
import { workloads } from './workloads.ts';

interface Baseline {
  readonly note: string;
  readonly recorded: { readonly node: string; readonly platform: string; readonly date: string };
  /** Keyed by `workload/mode`: that mode's time divided by the plain time. */
  readonly ratios: Record<string, number>;
}

const BASELINE = new URL('./baseline.json', import.meta.url);
const WARN = Number(process.env.BENCH_WARN ?? 1.15);
const FAIL = Number(process.env.BENCH_FAIL ?? 1.4);
const mode = process.argv.includes('--update')
  ? 'update'
  : process.argv.includes('--compare')
    ? 'compare'
    : 'print';

const tape = new Tape();
const fwd = new Forward();
let sink = 0;

const bench = new Bench({ time: Number(process.env.BENCH_TIME ?? 1500), warmupTime: 300 });
const nodes = new Map<string, number>();
for (const w of workloads) {
  tape.clear();
  w.f(tape, tape.inputs(w.at));
  nodes.set(w.name, tape.size);
  const grad = new Float64Array(w.at.length);
  const dir = w.at.map((_, i) => (i === 0 ? 1 : 0));
  bench
    .add(`${w.name}/plain`, () => {
      sink += w.f(plain, w.at);
    })
    .add(`${w.name}/record`, () => {
      tape.clear();
      sink += tape.value(w.f(tape, tape.inputs(w.at)));
    })
    .add(`${w.name}/record+backward`, () => {
      tape.clear();
      const xs = tape.inputs(w.at);
      tape.gradient(w.f(tape, xs), xs, grad);
      sink += grad[0]!;
    })
    .add(`${w.name}/forward`, () => {
      fwd.clear();
      sink += fwd.tangent(
        w.f(
          fwd,
          w.at.map((x, i) => fwd.input(x, dir[i]!)),
        ),
      );
    });
}

await bench.run();

const median = new Map<string, number>();
for (const task of bench.tasks) {
  const r = task.result;
  if (r.state !== 'completed') throw new Error(`${task.name}: ${r.state}`);
  median.set(task.name, r.latency.p50);
}

const ratios: Record<string, number> = {};
const rows: string[] = [];
for (const w of workloads) {
  const base = median.get(`${w.name}/plain`)!;
  const n = nodes.get(w.name)!;
  rows.push(`${w.name}: ${String(w.at.length)} inputs, ${String(n)} tape nodes`);
  for (const m of ['plain', 'record', 'record+backward', 'forward']) {
    const ms = median.get(`${w.name}/${m}`)!;
    if (m !== 'plain') ratios[`${w.name}/${m}`] = Number((ms / base).toFixed(3));
    rows.push(
      `  ${m.padEnd(16)} ${ms.toFixed(3).padStart(8)} ms  ${((ms * 1e6) / n).toFixed(1).padStart(6)} ns/node` +
        `  ${(ms / base).toFixed(2).padStart(6)}x plain`,
    );
  }
}
console.log(rows.join('\n'));
if (Number.isNaN(sink)) console.log('(sink is NaN)');

if (mode === 'update') {
  const baseline: Baseline = {
    note: 'Time of each mode as a multiple of plain evaluation, from `npm run bench:update`.',
    recorded: {
      node: process.version,
      platform: `${process.platform}-${process.arch}`,
      date: new Date().toISOString().slice(0, 10),
    },
    ratios,
  };
  writeFileSync(BASELINE, `${JSON.stringify(baseline, null, 2)}\n`);
  console.log(`\nwrote ${BASELINE.pathname}`);
} else if (mode === 'compare') {
  const baseline = JSON.parse(readFileSync(BASELINE, 'utf8')) as Baseline;
  let failed = false;
  console.log(
    `\ncompared with baseline from ${baseline.recorded.date} (${baseline.recorded.node}, ${baseline.recorded.platform}):`,
  );
  for (const [key, was] of Object.entries(baseline.ratios)) {
    const now = ratios[key];
    if (now === undefined) {
      console.log(`::warning::bench: ${key} is in the baseline but was not measured`);
      continue;
    }
    const change = now / was;
    const line = `${key.padEnd(32)} ${was.toFixed(2)}x -> ${now.toFixed(2)}x  (${change >= 1 ? '+' : ''}${((change - 1) * 100).toFixed(1)}%)`;
    if (change > FAIL) {
      failed = true;
      console.log(`::error::bench regression: ${line}`);
    } else if (change > WARN) {
      console.log(`::warning::bench slowdown: ${line}`);
    } else {
      console.log(`  ${line}`);
    }
  }
  if (failed) process.exitCode = 1;
}
