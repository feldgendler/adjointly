# Pitfalls

Each of these gives a wrong derivative without an error. Most are invisible to the type checker.
When a gradient looks wrong, run once with `new Tape({ debug: true })` (or `new Forward({ debug:
true })`): it checks every operand and throws at the first misuse it can see.

## Telling variables from numbers with `typeof`

**Never tell a variable from a plain number by `typeof`, or by its value.** A `Var` (or `Dual`) is a
number at run time, so `typeof x === 'number'` is true for every variable. Code that uses such a test
to turn "numbers" into constants turns every variable into a constant too, and every gradient comes
out exactly 0, silently:

```ts
import { type Arithmetic, gradient, Tape, type Var } from 'adjointly';

// Wrong: meant to lift plain numbers into constants and pass variables through.
function lift<T>(A: Arithmetic<T>, v: T | number): T {
  return typeof v === 'number' ? A.constant(v) : v;
}
gradient((A, [x]) => A.mul(lift(A, x), lift(A, 2)), [3]).gradient; //=> [0]

// Debug mode refuses a variable where a number belongs.
const t = new Tape({ debug: true });
const x: Var = t.input(3);
let message = '';
try {
  lift(t, x);
} catch (e) {
  message = (e as Error).message.split(',')[0]!;
}
message; //=> 'Tape.constant: argument 1 is a variable'
```

Keep variables and numbers in separate places (separate parameters, separate arrays, a field that
says which), and pass numbers to the plain-number operations, which take them as such:

```ts
import { gradient } from 'adjointly';

gradient((A, [x]) => A.mulc(x, 2), [3]).gradient; //=> [2]
```

## Arithmetic operators on a `Var`

`Var` and `Dual` are numbers at run time, and TypeScript allows `+`, `*`, `Math.exp` on number
subtypes. The result is arithmetic on _indices_: meaningless, and unchecked.

```ts
import { Tape } from 'adjointly';

const t = new Tape();
const x = t.input(10);
const y = t.input(20);
// Wrong: `x + y` adds the nodes' indices (1 and 2: slot 0 is reserved). Right:
t.value(t.add(x, y)); //=> 30
```

Route every operation through the arithmetic object, `Math.*` included.

## Reading a value out and feeding it back

`A.value(x)` is a plain number. Anything computed from it and fed back through `A.constant` is a
constant: the derivative through it is lost.

```ts
import { gradient } from 'adjointly';

// Intends x^2, differentiates as if the second x were a constant.
gradient((A, [x]) => A.mul(x, A.constant(A.value(x))), [3]).gradient; //=> [3]
gradient((A, [x]) => A.square(x), [3]).gradient; //=> [6]
```

Use `A.value` for branch conditions, logging, step sizes and results; never for a quantity a
derivative should flow through.

## Branches at their boundary, and kinks written by hand

The derivative is that of the branch taken. A hand-written `abs` is `+1` at zero; `A.abs` is `0`
by convention. Prefer the primitives for kinks.

```ts
import { gradient } from 'adjointly';

gradient((A, [x]) => (A.value(x) >= 0 ? x : A.neg(x)), [0]).gradient; //=> [1]
gradient((A, [x]) => A.abs(x), [0]).gradient; //=> [0]
```

A piecewise function that is discontinuous at a boundary has no derivative there, and the one you
get describes only the side taken. Finite differences across the boundary are meaningless too, so
test gradients away from boundaries.

## Unstable compositions

```ts
import { plain } from 'adjointly';

// Built by hand from exp and log, softplus overflows; the primitive does not.
plain.log(plain.add(plain.constant(1), plain.exp(800))); //=> Infinity
plain.softplus(800); //=> 800
```

Use `sigmoid`, `softplus`, `log1p`, `expm1`, `logsumexp` and `softmax` rather than assembling them.

## Leaving a function's domain

`log`, `sqrt`, `div` by zero and `pow` with a non-positive base produce `NaN` or `Infinity` in value
and derivative. A parameter that must stay positive should be reparametrised (`softplus(raw)`)
rather than clamped after the fact.

## Using a `Var` after `clear()` or `rewind()`

A `Var` is an index. After `clear()`, or `rewind(m)` to before it was made, it names nothing, or
whichever newer node took its slot. Nothing checks. Keep only `Var`s made before a `mark()` you
rewind to.

## A `Var` from another tape

The brand cannot tell two tapes apart. Passing a `Var` from tape A to tape B reads B's node with the
same index. Keep one tape per computation, and do not mix them.

## `noUncheckedIndexedAccess` and generic accumulators

Under that compiler option `xs[i]` is `T | undefined`. For a fixed number of inputs, type them as a
tuple (`[x, y]: readonly [T, T]`, or `ScalarFunction<[number, number]>`) and they are plain `T`s;
`tape.inputs([1, 2])` likewise returns a tuple of `Var`s. For a list of any length, index with `!`;
and note that `let s = xs[0]!` infers `NonNullable<T>`, which a later `s = A.add(s, ...)` (returning
`T`) cannot be assigned to. Annotate it: `let s: T = xs[0]!`.

## Adjoints after the output

`backward(output)` only zeroes and sweeps nodes up to `output`. Adjoints of nodes recorded after it
are left over from earlier sweeps and mean nothing.
