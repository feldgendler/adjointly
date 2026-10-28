/*
 * Evaluating schedules, and a first-improvement local search over the moves of schedule.ts that
 * tries candidates either in the order the gradient ranks them or in random order.
 */

import { Forward, plain, Tape, type Var } from '../../src/index.ts';
import { type Knobs, type Line, profit } from './model.ts';
import { apply, candidateMoves, estimate, flags, type Move, type Schedule } from './schedule.ts';

/** A seeded PRNG (mulberry32), uniform in [0, 1). */
export function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A line, its parameters, and counters of the work done on it. */
export class Problem {
  /** Full plain evaluations of a schedule's profit. */
  evaluations = 0;
  /** Forward-and-backward passes on the tape. */
  gradientPasses = 0;
  private readonly tape = new Tape();

  readonly line: Line;
  readonly knobs: Knobs<number>;

  constructor(line: Line, knobs: Knobs<number>) {
    this.line = line;
    this.knobs = knobs;
  }

  /** 1. The plain evaluation of a schedule. */
  profit(schedule: Schedule): number {
    this.evaluations++;
    return profit(plain, this.line, flags(this.line, schedule), this.knobs);
  }

  /**
   * 2. Profit and its gradient with respect to every station-minute's pause flag (relaxed to a
   * real number), from one forward pass recording the tape and one backward sweep.
   */
  gradient(pause: readonly number[]): { value: number; gradient: Float64Array } {
    this.gradientPasses++;
    const t = this.tape;
    t.clear();
    const ps = t.inputs(pause);
    const knobs = {
      price: t.constant(this.knobs.price),
      wearRate: t.constant(this.knobs.wearRate),
    };
    const y: Var = profit(t, this.line, ps, knobs);
    return { value: t.value(y), gradient: t.gradient(y, ps) };
  }

  /**
   * 5. d(profit)/d(knob) at a schedule, by forward mode: one pass with that knob seeded.
   */
  sensitivity(
    schedule: Schedule,
    knob: keyof Knobs<number>,
  ): { value: number; derivative: number } {
    const F = new Forward();
    const pause = flags(this.line, schedule).map((p) => F.constant(p));
    const knobs = {
      price: F.input(this.knobs.price, knob === 'price' ? 1 : 0),
      wearRate: F.input(this.knobs.wearRate, knob === 'wearRate' ? 1 : 0),
    };
    const y = profit(F, this.line, pause, knobs);
    return { value: F.value(y), derivative: F.tangent(y) };
  }
}

/**
 * What a gradient pass (one recording and one backward sweep) costs, in plain evaluations of the
 * same schedule, measured here by timing both, interleaved, `rounds` times each. The fastest time
 * of each is used: on a busy machine the slower runs measure the machine, not the code.
 */
export function measurePassCost(
  line: Line,
  knobs: Knobs<number>,
  schedule: Schedule,
  rounds = 25,
): number {
  const p = new Problem(line, knobs); // its counters are not the searches'
  const pause = flags(line, schedule);
  let plainBest = Infinity;
  let passBest = Infinity;
  for (let r = 0; r < rounds; r++) {
    const t0 = performance.now();
    p.profit(schedule);
    const t1 = performance.now();
    p.gradient(pause);
    const t2 = performance.now();
    plainBest = Math.min(plainBest, t1 - t0);
    passBest = Math.min(passBest, t2 - t1);
  }
  return passBest / plainBest;
}

export interface RankedMove {
  readonly move: Move;
  /** First-order estimate of the change in profit. */
  readonly estimate: number;
}

/** 3. Every candidate move, with its first-order estimate, best first. */
export function rankMoves(
  problem: Problem,
  schedule: Schedule,
  gradient: ArrayLike<number>,
  maxShift: number,
  maxTransfer: number,
): RankedMove[] {
  return candidateMoves(problem.line, schedule, maxShift, maxTransfer)
    .map((move) => ({ move, estimate: estimate(problem.line, schedule, move, gradient) }))
    .sort((a, b) => b.estimate - a.estimate);
}

export interface SearchResult {
  readonly schedule: Schedule;
  readonly profit: number;
  readonly improvements: number;
  /** Full evaluations including the final round that found no improving move. */
  readonly evaluations: number;
  readonly gradientPasses: number;
  /** Profit after each improvement, with the full evaluations and gradient passes made by then. */
  readonly trace: readonly {
    readonly evaluations: number;
    readonly gradientPasses: number;
    readonly profit: number;
  }[];
}

export interface SearchOptions {
  readonly order: 'gradient' | 'random';
  readonly seed: number;
  readonly maxShift: number;
  readonly maxTransfer: number;
}

/**
 * The work a search had done when its profit first reached `profit`, if it did, in evaluation
 * equivalents: full evaluations plus gradient passes, each counted as `passCost` evaluations.
 */
export function workToReach(
  result: SearchResult,
  profit: number,
  passCost: number,
): number | undefined {
  const at = result.trace.find((p) => p.profit >= profit - 1e-9);
  return at && at.evaluations + passCost * at.gradientPasses;
}

/** All the work a search did, the final round that found nothing included, as in `workToReach`. */
export function totalWork(result: SearchResult, passCost: number): number {
  return result.evaluations + passCost * result.gradientPasses;
}

/**
 * 4. First-improvement local search: from the current schedule, evaluate candidate moves one by one
 * in the chosen order and accept the first that improves profit; stop when none does. In gradient
 * order the gradient is recomputed (one tape pass) after every accepted move.
 */
export function localSearch(
  problem: Problem,
  start: Schedule,
  options: SearchOptions,
): SearchResult {
  const random = rng(options.seed);
  const startEvaluations = problem.evaluations;
  const startPasses = problem.gradientPasses;
  const spent = (): number => problem.evaluations - startEvaluations;
  const passes = (): number => problem.gradientPasses - startPasses;
  let schedule = start;
  let best = problem.profit(schedule);
  let improvements = 0;
  const trace = [{ evaluations: spent(), gradientPasses: passes(), profit: best }];

  for (;;) {
    let moves: Move[];
    if (options.order === 'gradient') {
      const { gradient } = problem.gradient(flags(problem.line, schedule));
      moves = rankMoves(problem, schedule, gradient, options.maxShift, options.maxTransfer).map(
        (r) => r.move,
      );
    } else {
      moves = candidateMoves(problem.line, schedule, options.maxShift, options.maxTransfer);
      for (let i = moves.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [moves[i], moves[j]] = [moves[j]!, moves[i]!];
      }
    }
    let improved = false;
    for (const move of moves) {
      const next = apply(schedule, move);
      const value = problem.profit(next);
      if (value > best + 1e-9) {
        schedule = next;
        best = value;
        improvements++;
        trace.push({ evaluations: spent(), gradientPasses: passes(), profit: best });
        improved = true;
        break;
      }
    }
    if (!improved) break;
  }

  return {
    schedule,
    profit: best,
    improvements,
    evaluations: spent(),
    gradientPasses: passes(),
    trace,
  };
}
