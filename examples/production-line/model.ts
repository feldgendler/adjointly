/*
 * The production line: a few stations in series with buffers between them, simulated minute by
 * minute over a shift, written once against Arithmetic so the same code gives the profit of a
 * schedule (plain), its gradient with respect to every minute's maintenance flag (Tape), and its
 * sensitivity to a parameter (Forward).
 *
 * Every minute, for each station s (upstream first):
 *
 *   run       = 1 - pause[s][t]                       pause is 1 during maintenance, 0 otherwise
 *   starve    = sigmoid(k * (buffer before s - 0.5))  1 when parts are waiting, ~0 when none are
 *   block     = sigmoid(k * (room after s - 0.5))     1 when the next buffer has room
 *   made      = capacity * run * starve * block       parts processed this minute
 *   yield     = sigmoid(yieldAt0 - yieldSlope * wear) the good fraction falls as wear builds up
 *   wear      = max(0, wear + wearRate * run - recovery * pause), the max smoothed
 *
 * The good parts move on to the next buffer (or, from the last station, are sold); the rest are
 * scrapped. The first station draws from a raw-material buffer that deliveries refill at a rate
 * that varies over the shift, with shortages when it drops. Revenue is the price times parts sold, each
 * weighted by what a part finished in that minute fetches; profit is revenue minus a cost per
 * maintenance minute and a cost per scrapped part.
 *
 * It is a smooth fluid model: parts are real numbers, and a starved station's sigmoid lets a
 * buffer dip a few hundredths below zero rather than stopping at exactly zero. That is what makes
 * profit differentiable in the pause flags, and why a flag can be relaxed from {0, 1} to [0, 1].
 */

import type { Arithmetic } from '../../src/index.ts';

export interface Line {
  /** Number of stations in series. */
  readonly stations: number;
  /** Minutes in the shift. */
  readonly minutes: number;
  /** Parts a station can process per minute. */
  readonly capacity: readonly number[];
  /** Buffer capacity between stations (the raw-material buffer is unbounded). */
  readonly bufferSize: number;
  /** Parts in each buffer, raw material included, at the start of the shift. */
  readonly initialBuffer: number;
  /** Raw material delivered in each minute of the shift. */
  readonly supply: readonly number[];
  /** What a part finished in each minute fetches, as a multiple of the price. */
  readonly value: readonly number[];
  /** Steepness of the starve/block sigmoids. */
  readonly steepness: number;
  /** Wear at the start of the shift, per station. */
  readonly initialWear: readonly number[];
  /** Wear removed per minute of maintenance (wear never goes below zero). */
  readonly recovery: number;
  /** yield = sigmoid(yieldAt0 - yieldSlope * wear). */
  readonly yieldAt0: number;
  readonly yieldSlope: number;
  /** Cost of a station-minute of maintenance. */
  readonly maintenanceCost: number;
  /** Cost of a scrapped part. */
  readonly scrapCost: number;
}

/** The parameters the example differentiates with respect to, besides the pause flags. */
export interface Knobs<T> {
  /** Sale price of a good part. */
  readonly price: T;
  /** Wear added per minute of running. */
  readonly wearRate: T;
}

/** `usual` in every minute, except `during` within the given intervals. */
export function profile(
  minutes: number,
  usual: number,
  during: number,
  intervals: readonly (readonly [number, number])[],
): number[] {
  return Array.from({ length: minutes }, (_, t) =>
    intervals.some(([a, b]) => t >= a && t < b) ? during : usual,
  );
}

export const defaultLine: Line = {
  stations: 3,
  minutes: 480,
  // Deliveries run short twice in the shift.
  supply: profile(480, 1.05, 0.1, [
    [130, 170],
    [330, 360],
  ]),
  // Parts finished while the packing line downstream is down go to stock at a discount.
  value: profile(480, 1, 0.4, [
    [200, 250],
    [400, 440],
  ]),
  capacity: [1.15, 1.0, 1.1],
  bufferSize: 12,
  initialBuffer: 3,
  steepness: 6,
  initialWear: [0.4, 0.8, 0.2],
  recovery: 0.015,
  yieldAt0: 5,
  yieldSlope: 2.2,
  maintenanceCost: 0.5,
  scrapCost: 2,
};

export const defaultKnobs: Knobs<number> = { price: 10, wearRate: 0.006 };

/**
 * Profit over the shift. `pause[s * line.minutes + t]` is station s's maintenance flag in minute
 * t: 1 paused, 0 running, anything between for the relaxed problem.
 */
export function profit<T>(A: Arithmetic<T>, line: Line, pause: readonly T[], knobs: Knobs<T>): T {
  const { stations, minutes } = line;
  const one = A.constant(1);
  const k = A.constant(line.steepness);
  const half = A.constant(0.5);
  const recovery = A.constant(line.recovery);
  const yieldAt0 = A.constant(line.yieldAt0);
  const yieldSlope = A.constant(line.yieldSlope);
  const room = A.constant(line.bufferSize);
  const capacity = line.capacity.map((c) => A.constant(c));

  // buffer[s] holds parts waiting in front of station s; buffer[0] is raw material.
  const buffer: T[] = Array.from({ length: stations }, () => A.constant(line.initialBuffer));
  const wear: T[] = line.initialWear.map((w) => A.constant(w));
  let sold = A.constant(0);
  let scrapped = A.constant(0);
  let paused = A.constant(0);

  for (let t = 0; t < minutes; t++) {
    buffer[0] = A.add(buffer[0]!, A.constant(line.supply[t]!));
    for (let s = 0; s < stations; s++) {
      const p = pause[s * minutes + t]!;
      const run = A.sub(one, p);
      let made = A.mul(capacity[s]!, run);
      made = A.mul(made, A.sigmoid(A.mul(k, A.sub(buffer[s]!, half))));
      if (s < stations - 1) {
        const space = A.sub(A.sub(room, buffer[s + 1]!), half);
        made = A.mul(made, A.sigmoid(A.mul(k, space)));
      }
      const good = A.mul(made, A.sigmoid(A.sub(yieldAt0, A.mul(yieldSlope, wear[s]!))));

      buffer[s] = A.sub(buffer[s]!, made);
      if (s < stations - 1) buffer[s + 1] = A.add(buffer[s + 1]!, good);
      else sold = A.add(sold, A.mul(A.constant(line.value[t]!), good));
      scrapped = A.add(scrapped, A.sub(made, good));
      paused = A.add(paused, p);

      // Wear builds while running and falls by a fixed amount per minute of maintenance,
      // floored smoothly at zero: softplus(k x) / k is max(x, 0) with the corner rounded off.
      const w = A.sub(A.add(wear[s]!, A.mul(knobs.wearRate, run)), A.mul(recovery, p));
      wear[s] = A.div(A.softplus(A.mul(k, w)), k);
    }
  }

  const revenue = A.mul(knobs.price, sold);
  const costs = A.add(
    A.mul(A.constant(line.maintenanceCost), paused),
    A.mul(A.constant(line.scrapCost), scrapped),
  );
  return A.sub(revenue, costs);
}
