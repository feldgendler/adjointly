# Pitfalls

Each of these gives a wrong derivative without an error. Most are invisible to the type checker.

## Arithmetic operators on a `Var`

`Var` and `Dual` are numbers at run time, and TypeScript allows `+`, `*`, `Math.exp` on number
subtypes. The result is arithmetic on _indices_: meaningless, and unchecked.

```ts
import { Tape } from 'adjointly';

const t = new Tape();
const x = t.input(10);
const y = t.input(20);
// Wrong: `x + y` adds the indices 0 and 1. Right:
t.value(t.add(x, y)); //=> 30
```

Route every operation through the arithmetic object, `Math.*` included.

## Reading a value out and feeding it back

`A.value(x)` is a plain number. Anything computed from it and fed back through `A.constant` is a
constant: the derivative through it is lost.

```ts
import { gradient } from 'adjointly';

// Intends x^2, differentiates as if the second x were a constant.
gradient((A, [x]) => A.mul(x!, A.constant(A.value(x!))), [3]).gradient; //=> [3]
gradient((A, [x]) => A.square(x!), [3]).gradient; //=> [6]
```

Use `A.value` for branch conditions, logging, step sizes and results; never for a quantity a
derivative should flow through.

## Branches at their boundary, and kinks written by hand

The derivative is that of the branch taken. A hand-written `abs` is `+1` at zero; `A.abs` is `0`
by convention. Prefer the primitives for kinks.

```ts
import { gradient } from 'adjointly';

gradient((A, [x]) => (A.value(x!) >= 0 ? x! : A.neg(x!)), [0]).gradient; //=> [1]
gradient((A, [x]) => A.abs(x!), [0]).gradient; //=> [0]
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

Under that compiler option `xs[i]` is `T | undefined`, so differentiable code indexes with `!`; and
`let s = xs[0]!` infers `NonNullable<T>`, which a later `s = A.add(s, ...)` (returning `T`) cannot
be assigned to. Annotate it: `let s: T = xs[0]!`.

## Adjoints after the output

`backward(output)` only zeroes and sweeps nodes up to `output`. Adjoints of nodes recorded after it
are left over from earlier sweeps and mean nothing.
