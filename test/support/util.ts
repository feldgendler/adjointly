/* Shared test helpers. */

import { expect } from 'vitest';

/** A small seeded PRNG (mulberry32), uniform in [0, 1). */
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

/** Assert `|actual - expected| <= atol + rtol * |expected|`, with a readable message. */
export function expectClose(actual: number, expected: number, rtol = 1e-12, atol = 1e-15): void {
  if (Object.is(actual, expected)) return;
  const ok = Math.abs(actual - expected) <= atol + rtol * Math.abs(expected);
  expect(ok, `${String(actual)} is not within rtol ${String(rtol)} of ${String(expected)}`).toBe(
    true,
  );
}

/** {@link expectClose} elementwise. */
export function expectAllClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  rtol = 1e-12,
  atol = 1e-15,
): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) expectClose(actual[i]!, expected[i]!, rtol, atol);
}
