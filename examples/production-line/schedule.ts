/*
 * The discrete decisions: maintenance windows, and the moves that change them while keeping the
 * total minutes of maintenance in the shift fixed.
 */

import type { Line } from './model.ts';

/** A maintenance window: station `station` is paused for minutes [start, start + length). */
export interface Window {
  readonly station: number;
  readonly start: number;
  readonly length: number;
}

export type Schedule = readonly Window[];

/** The shortest window worth stopping a station for. */
export const MIN_LENGTH = 5;

/** The per-minute pause flags of a schedule, laid out as `profit` expects. */
export function flags(line: Line, schedule: Schedule): number[] {
  const out = new Array<number>(line.stations * line.minutes).fill(0);
  for (const w of schedule) {
    for (let t = w.start; t < w.start + w.length; t++) out[w.station * line.minutes + t] = 1;
  }
  return out;
}

/** Whether every window fits in the shift, is long enough, and overlaps no other on its station. */
export function feasible(line: Line, schedule: Schedule): boolean {
  for (const [i, w] of schedule.entries()) {
    if (w.length < MIN_LENGTH || w.start < 0 || w.start + w.length > line.minutes) return false;
    for (const v of schedule.slice(i + 1)) {
      if (v.station === w.station && w.start < v.start + v.length && v.start < w.start + w.length) {
        return false;
      }
    }
  }
  return true;
}

export type Move =
  | { readonly kind: 'shift'; readonly window: number; readonly by: number }
  | {
      readonly kind: 'transfer';
      readonly from: number;
      readonly to: number;
      readonly minutes: number;
    };

/** The schedule after a move (which may be infeasible; check with `feasible`). */
export function apply(schedule: Schedule, move: Move): Window[] {
  const next = schedule.map((w) => ({ ...w }));
  if (move.kind === 'shift') {
    const w = next[move.window]!;
    next[move.window] = { ...w, start: w.start + move.by };
  } else {
    // Minutes leave the end of one window and join the end of another.
    const from = next[move.from]!;
    const to = next[move.to]!;
    next[move.from] = { ...from, length: from.length - move.minutes };
    next[move.to] = { ...to, length: to.length + move.minutes };
  }
  return next;
}

/** Shifts by 1..maxShift minutes either way; transfers of 1..maxTransfer minutes between windows. */
export function candidateMoves(
  line: Line,
  schedule: Schedule,
  maxShift: number,
  maxTransfer: number,
): Move[] {
  const moves: Move[] = [];
  for (let i = 0; i < schedule.length; i++) {
    for (let by = -maxShift; by <= maxShift; by++) {
      if (by !== 0) moves.push({ kind: 'shift', window: i, by });
    }
    for (let j = 0; j < schedule.length; j++) {
      if (j === i) continue;
      for (let m = 1; m <= maxTransfer; m++)
        moves.push({ kind: 'transfer', from: i, to: j, minutes: m });
    }
  }
  return moves.filter((m) => feasible(line, apply(schedule, m)));
}

/**
 * The first-order estimate of a move's effect on profit: the gradient with respect to the pause
 * flags, dotted with the change the move makes to them. Only the changed minutes contribute.
 */
export function estimate(
  line: Line,
  schedule: Schedule,
  move: Move,
  gradient: ArrayLike<number>,
): number {
  const before = flags(line, schedule);
  const after = flags(line, apply(schedule, move));
  let delta = 0;
  for (let i = 0; i < before.length; i++) {
    const d = after[i]! - before[i]!;
    if (d !== 0) delta += gradient[i]! * d;
  }
  return delta;
}

export function describe(move: Move): string {
  return move.kind === 'shift'
    ? `shift window ${String(move.window)} by ${move.by > 0 ? '+' : ''}${String(move.by)} min`
    : `move ${String(move.minutes)} min from window ${String(move.from)} to window ${String(move.to)}`;
}
