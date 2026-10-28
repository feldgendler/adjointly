/*
 * The opt-in debug mode of Tape and Forward (`{ debug: true }`).
 *
 * A Var or Dual is a plain number at run time, the index of its slot, so nothing stops a raw
 * number, a stale handle or another tape's handle from being used as one: it silently reads
 * whatever slot has that index. In debug mode the handles a tape gives out are instead
 * `epoch * 2^31 + index`, where the epoch is unique to the tape and to the stretch of recording
 * since its last clear() or rewind(), and every method checks and decodes its operands before
 * running the ordinary code. Each slot remembers the epoch it was written in, so a handle whose
 * slot has since been discarded, or reused, is caught too.
 *
 * The checks are installed as own properties of the debug instance, wrapping the class's methods;
 * the class itself is untouched, so a tape without `debug` runs exactly the code it ran before.
 */

import { constantOperandOps } from './arithmetic.ts';
import { binaryNames, unaryNames } from './primitives.ts';

/** Indices fit in 31 bits (parents are stored in an Int32Array). */
const STRIDE = 2 ** 31;
/** Epochs run from 1 to EPOCHS - 1, so that a handle stays below 2^52, an exact integer. */
const EPOCHS = 2 ** 21;

let lastEpoch = 0;

/**
 * A fresh epoch, unique among all debug tapes and Forwards until 2^21 - 1 epochs have been used,
 * after which numbering wraps around; a stale handle could then be mistaken for a live one only if
 * it were used exactly that many clears later, at the same index.
 */
function nextEpoch(): number {
  lastEpoch = (lastEpoch % (EPOCHS - 1)) + 1;
  return lastEpoch;
}

/** The handles of one debug Tape or Forward: issuing them and checking them. */
class Handles {
  private epoch = 0;
  /** Every epoch this instance has used, to tell its own stale handles from another's. */
  private readonly own = new Set<number>();
  /** The epoch each slot was last written in. */
  private slotEpoch = new Int32Array(64);

  private readonly kind: string;
  private readonly first: number;
  private readonly target: { readonly size: number };

  constructor(kind: string, first: number, target: { readonly size: number }) {
    this.kind = kind;
    this.first = first;
    this.target = target;
    this.begin();
  }

  /** Start a new epoch: handles issued from now on are told apart from earlier ones. */
  begin(): void {
    this.epoch = nextEpoch();
    this.own.add(this.epoch);
  }

  /** The handle for the slot `index` just written. */
  issue(index: number): number {
    while (index >= this.slotEpoch.length) {
      const grown = new Int32Array(this.slotEpoch.length * 2);
      grown.set(this.slotEpoch);
      this.slotEpoch = grown;
    }
    this.slotEpoch[index] = this.epoch;
    return this.epoch * STRIDE + index;
  }

  /**
   * Throw if `value`, given where a plain number belongs, is one of this instance's variables: what
   * a variable mistaken for a number looks like, typically after a `typeof x === 'number'` test,
   * which every variable passes. Only this instance's handles are recognised, so that no ordinary
   * number is ever refused: a handle is a whole number of 2^31 or more, encoding an epoch of this
   * instance and a slot written in that epoch.
   */
  checkNumber(value: unknown, where: string): void {
    if (typeof value !== 'number' || !Number.isSafeInteger(value)) return;
    const epoch = Math.floor(value / STRIDE);
    const index = value - epoch * STRIDE;
    if (this.own.has(epoch) && this.slotEpoch[index] === epoch) {
      throw new RangeError(
        `${where} is a variable, where a plain number belongs. A variable is a number at run time, ` +
          "so typeof x === 'number' is true for it too: keep variables and numbers apart instead.",
      );
    }
  }

  /** The slot index of `handle`, or a `RangeError` saying what is wrong with it. */
  resolve(handle: unknown, where: string): number {
    if (typeof handle !== 'number' || !Number.isSafeInteger(handle) || handle < 0) {
      throw new RangeError(`${where} is not a variable (it is ${String(handle)})`);
    }
    const epoch = Math.floor(handle / STRIDE);
    const index = handle - epoch * STRIDE;
    if (epoch === 0) {
      throw new RangeError(
        `${where} is the plain number ${String(handle)}, not a variable of this ${this.kind}. ` +
          'Numbers enter through constant() or input(), or as the number operand of addc(), ' +
          'mulc() and the like. Do not tell variables from numbers with typeof: both are numbers.',
      );
    }
    if (!this.own.has(epoch)) {
      throw new RangeError(`${where} is a variable of another Tape or Forward`);
    }
    if (
      index < this.first ||
      index >= this.first + this.target.size ||
      this.slotEpoch[index] !== epoch
    ) {
      throw new RangeError(`${where} is stale: its node was discarded by clear() or rewind()`);
    }
    return index;
  }
}

/**
 * How a method's arguments and result are treated: a variable, a list of them, a plain number, a
 * list of them, or anything (passed through).
 */
type Arg = 'v' | 'vs' | 'n' | 'ns' | '-';
type Signature = readonly [readonly Arg[], Arg];

const arithmetic: Record<string, Signature> = {
  constant: [['n'], 'v'],
  value: [['v'], '-'],
};
for (const name of unaryNames) arithmetic[name] = [['v'], 'v'];
for (const name of binaryNames) arithmetic[name] = [['v', 'v'], 'v'];
for (const [name, { constantFirst }] of Object.entries(constantOperandOps)) {
  arithmetic[name] = [constantFirst ? ['n', 'v'] : ['v', 'n'], 'v'];
}

const signatures: Record<'Tape' | 'Forward', Record<string, Signature>> = {
  Tape: {
    ...arithmetic,
    input: [['n'], 'v'],
    inputs: [['ns'], 'vs'],
    backward: [['v', 'n'], '-'],
    vjp: [['vs', '-'], '-'],
    adjoint: [['v'], '-'],
    adjoints: [['vs', '-'], '-'],
    gradient: [['v', 'vs', '-'], '-'],
  },
  Forward: {
    ...arithmetic,
    input: [['n', 'n'], 'v'],
    tangent: [['v'], '-'],
  },
};

type Method = (...args: unknown[]) => unknown;

/**
 * Turn `target`, a freshly constructed Tape or Forward, into its debug version. `first` is the
 * index of its first slot (1 for a Tape, whose slot 0 is reserved; 0 for a Forward).
 */
export function installChecks(
  target: { readonly size: number },
  kind: 'Tape' | 'Forward',
  first: number,
): void {
  const handles = new Handles(kind, first, target);
  const self = target as unknown as Record<string, Method>;
  const proto = Object.getPrototypeOf(target) as Record<string, Method>;

  const element = (where: string, k: number): string => `${where}, element ${String(k + 1)}`;
  const decode = (arg: Arg, value: unknown, where: string): unknown => {
    switch (arg) {
      case 'v':
        return handles.resolve(value, where);
      case 'vs':
        return (value as readonly unknown[]).map((h, k) => handles.resolve(h, element(where, k)));
      case 'n':
        handles.checkNumber(value, where);
        return value;
      case 'ns':
        Array.from(value as ArrayLike<unknown>, (x, k) => {
          handles.checkNumber(x, element(where, k));
        });
        return value;
      case '-':
        return value;
    }
  };
  const encode = (arg: Arg, value: unknown): unknown =>
    arg === 'v'
      ? handles.issue(value as number)
      : arg === 'vs'
        ? (value as number[]).map((i) => handles.issue(i))
        : value;

  for (const [name, [args, result]] of Object.entries(signatures[kind])) {
    // gradient() calls backward() and adjoints() on `this`, which would then see decoded
    // indices; the decoded call goes to the class's own methods instead.
    const run: Method =
      name === 'gradient'
        ? (output, inputs, out) => {
            proto.backward!.call(target, output);
            return proto.adjoints!.call(target, inputs, out);
          }
        : (...decoded) => proto[name]!.apply(target, decoded);
    self[name] = (...params: unknown[]) => {
      const decoded = params.map((p, i) =>
        decode(args[i] ?? '-', p, `${kind}.${name}: argument ${String(i + 1)}`),
      );
      return encode(result, run(...decoded));
    };
  }
  // Every clear() and rewind() starts a new epoch, retiring the handles of discarded slots.
  self.clear = () => {
    proto.clear!.call(target);
    handles.begin();
  };
  if (kind === 'Tape') {
    self.rewind = (mark) => {
      proto.rewind!.call(target, mark);
      handles.begin();
    };
  }
}
