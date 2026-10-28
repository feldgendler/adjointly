/*
 * The `capacity` option shared by Tape and Forward.
 */

/** `capacity` if it is a positive integer, the default if it is not given; otherwise throws. */
export function initialCapacity(capacity: number | undefined, owner: string): number {
  if (capacity === undefined) return 1024;
  if (!Number.isInteger(capacity) || capacity < 1) {
    throw new RangeError(`${owner}: capacity must be a positive integer, not ${String(capacity)}`);
  }
  return capacity;
}
