/*
 * Times every workload in four modes: evaluated plainly, recorded on a tape, recorded and swept
 * backwards, and in forward mode along one direction. Not part of the tests.
 *
 *   npm run bench                                  this checkout, as a table
 *   node bench/measure.ts --lib <dir>/src/index.ts --json
 *                                                  another checkout's library, as JSON (ab.ts)
 *
 * BENCH_TIME sets the milliseconds tinybench spends on each case (default 1000); BENCH_ONLY, a
 * substring of workload names, limits the run to the workloads whose name contains it.
 */

import { pathToFileURL } from 'node:url';
import { Bench } from 'tinybench';
import { type Library, workloads } from './workloads.ts';

export const MODES = ['plain', 'record', 'record+backward', 'forward'] as const;

/** A case's median time in milliseconds, and the tape nodes its workload records. */
export interface Timing {
  readonly ms: number;
  readonly nodes: number;
}

/** Median time of each `workload/mode` case. */
export async function measure(lib: Library, time: number): Promise<Record<string, Timing>> {
  const tape = new lib.Tape();
  const fwd = new lib.Forward();
  let sink = 0;
  const bench = new Bench({ time, warmupTime: Math.min(300, time) });
  const nodes = new Map<string, number>();
  const only = process.env.BENCH_ONLY ?? '';
  for (const w of workloads(lib).filter((w) => w.name.includes(only))) {
    tape.clear();
    w.f(tape, tape.inputs(w.at));
    nodes.set(w.name, tape.size);
    const grad = new Float64Array(w.at.length);
    const dir = w.at.map((_, i) => (i === 0 ? 1 : 0));
    bench
      .add(`${w.name}/plain`, () => {
        sink += w.f(lib.plain, w.at);
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
  if (Number.isNaN(sink)) throw new Error('a workload produced NaN');
  const out: Record<string, Timing> = {};
  for (const task of bench.tasks) {
    const r = task.result;
    if (r.state !== 'completed') throw new Error(`${task.name}: ${r.state}`);
    out[task.name] = { ms: r.latency.p50, nodes: nodes.get(task.name.split('/')[0]!)! };
  }
  return out;
}

if (import.meta.url === pathToFileURL(process.argv[1]!).href) {
  const at = process.argv.indexOf('--lib');
  const path =
    at >= 0 ? process.argv[at + 1]! : new URL('../src/index.ts', import.meta.url).pathname;
  const lib = (await import(pathToFileURL(path).href)) as Library;
  const timings = await measure(lib, Number(process.env.BENCH_TIME ?? 1000));
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(timings));
  } else {
    const names = [...new Set(Object.keys(timings).map((k) => k.split('/')[0]!))];
    for (const name of names) {
      const plain = timings[`${name}/plain`]!;
      console.log(`${name}: ${String(plain.nodes)} tape nodes`);
      for (const mode of MODES) {
        const t = timings[`${name}/${mode}`]!;
        console.log(
          `  ${mode.padEnd(16)} ${t.ms.toFixed(3).padStart(8)} ms ` +
            `${((t.ms * 1e6) / t.nodes).toFixed(1).padStart(6)} ns/node ` +
            `${(t.ms / plain.ms).toFixed(2).padStart(6)}x plain`,
        );
      }
    }
  }
}
