/*
 * A/B benchmark: this checkout ("head") against another ("base"), on the same machine, in
 * alternating rounds, so that machine speed and load cancel out. CI runs it on every pull request
 * with the PR's base commit, and on every push to main with the commit main was at before.
 *
 *   node bench/ab.ts --base <dir> [--head <dir>] [--rounds 5]
 *
 * Each round runs bench/measure.ts (this checkout's copy, so both sides time the same workloads)
 * once against each side's src/, as separate processes, alternating which side goes first. A
 * case's time on each side is the median over the rounds; the verdict is head / base. A ratio
 * above BENCH_WARN (default 1.10) is reported as a warning annotation, above BENCH_FAIL (default
 * 1.25) as an error, and the script then exits 1.
 */

import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import type { Timing } from './measure.ts';

const arg = (name: string, fallback?: string): string => {
  const i = process.argv.indexOf(name);
  const value = i >= 0 ? process.argv[i + 1] : fallback;
  if (value === undefined) throw new Error(`missing ${name}`);
  return value;
};

const here = resolve(import.meta.dirname, '..');
const base = resolve(arg('--base'));
const head = resolve(arg('--head', here));
const rounds = Number(arg('--rounds', '5'));
const WARN = Number(process.env.BENCH_WARN ?? 1.1);
const FAIL = Number(process.env.BENCH_FAIL ?? 1.25);

function run(dir: string): Record<string, Timing> {
  const out = execFileSync(
    process.execPath,
    [join(here, 'bench', 'measure.ts'), '--lib', join(dir, 'src', 'index.ts'), '--json'],
    { encoding: 'utf8', env: { ...process.env, BENCH_TIME: process.env.BENCH_TIME ?? '600' } },
  );
  return JSON.parse(out.trim().split('\n').at(-1)!) as Record<string, Timing>;
}

const samples = { base: [] as Record<string, Timing>[], head: [] as Record<string, Timing>[] };
for (let r = 0; r < rounds; r++) {
  const order = r % 2 === 0 ? (['base', 'head'] as const) : (['head', 'base'] as const);
  for (const side of order) samples[side].push(run(side === 'base' ? base : head));
  console.log(`round ${String(r + 1)} of ${String(rounds)} done`);
}

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
};

let failed = false;
console.log(`\nhead ${head}\nbase ${base}\nmedian of ${String(rounds)} rounds each:`);
for (const key of Object.keys(samples.head[0]!)) {
  const h = median(samples.head.map((s) => s[key]!.ms));
  if (!samples.base.every((s) => key in s)) {
    console.log(`  ${key.padEnd(32)} head ${h.toFixed(3)} ms (no such case on base)`);
    continue;
  }
  const b = median(samples.base.map((s) => s[key]!.ms));
  const ratio = h / b;
  const line = `${key.padEnd(32)} base ${b.toFixed(3)} ms, head ${h.toFixed(3)} ms: ${ratio.toFixed(3)}x`;
  if (ratio > FAIL) {
    failed = true;
    console.log(`::error::bench regression: ${line}`);
  } else if (ratio > WARN) {
    console.log(`::warning::bench slowdown: ${line}`);
  } else {
    console.log(`  ${line}`);
  }
}
if (failed) process.exitCode = 1;
