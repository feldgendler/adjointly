# adjointly

Automatic differentiation for scalar `float64` programs in TypeScript. Write a function once,
against a small arithmetic interface, and get from the same code:

- its **value**, on plain numbers, at full speed;
- its **gradient** with respect to every input, by **reverse mode** (one backward sweep over a
  tape stored in typed arrays, whatever the number of inputs);
- its **directional derivative** along any direction, by **forward mode**.

Derivatives are exact to floating-point rounding, not finite-difference approximations. Ordinary
control flow (loops, branches, early returns) just works. No runtime dependencies, ESM only,
Node 22 or later (or any modern browser or bundler).

> **License:** not yet decided. The package is `UNLICENSED` and private until it is.

## Quick start

```ts
import { type Arithmetic, evaluate, gradient, jvp } from 'adjointly';

// f(x, y) = sin(x) * e^y, written once against the arithmetic interface
function f<T>(A: Arithmetic<T>, [x, y]: readonly T[]): T {
  return A.mul(A.sin(x!), A.exp(y!));
}

evaluate(f, [1, 0]); //=> sin(1)
gradient(f, [1, 0]); //=> { value: sin(1), gradient: [cos(1), sin(1)] }
jvp(f, [1, 0], [1, 1]); //=> { value: sin(1), derivative: cos(1) + sin(1) }
```

(The `!`s are only there because this repository compiles with `noUncheckedIndexedAccess`.)

## Concepts

**The arithmetic interface.** A differentiable function takes an `Arithmetic<T>` (here `A`) and
values of an opaque type `T`, and does every operation through `A`: `A.add(x, y)`, `A.exp(x)`.
`T` is `number` when the function is evaluated plainly, and an index into storage when it is being
differentiated. Numbers enter through `A.constant(c)`; values leave through `A.value(x)`.

**The tape (reverse mode).** A `Tape` records each operation as a node: its value, the one or two
nodes it was computed from, and the _local partial derivatives_ with respect to them, computed on
the spot (for `y = x * z` those are `z` and `x`). That forward pass is the evaluation itself.

**Adjoints.** The adjoint of a node is the derivative of the output with respect to that node.
The output's own adjoint is 1. Walking the tape backwards, each node adds _partial × its own
adjoint_ into each parent's adjoint (the chain rule). When the walk reaches the inputs, their
adjoints are the gradient. One backward sweep costs a small constant times one evaluation, no
matter how many inputs there are.

**Forward mode.** `Forward` instead carries a _tangent_ beside every value: the derivative along
one direction `v` fixed when the inputs are created. Each operation computes its tangent from its
arguments' tangents as it runs, so one pass gives `∇f · v` with no tape to walk back.

**Which to use.**

| You have                                       | You want                                    | Use                              |
| ---------------------------------------------- | ------------------------------------------- | -------------------------------- |
| many inputs, one output (a loss, a likelihood) | the gradient                                | reverse mode: `gradient`, `Tape` |
| few inputs, or one direction                   | a directional derivative, a Jacobian column | forward mode: `jvp`, `Forward`   |
| several outputs                                | a weighted combination of their gradients   | `Tape.vjp`                       |
| no derivative                                  | the value, fast                             | `evaluate`, `plain`              |

A full gradient by forward mode costs one pass per input; by reverse mode, one pass and one sweep.

**Control flow.** Branch on values with `A.value(x)`, loop as long as you like, return early.
The tape records the operations that actually ran, so the derivative is that of the branch taken:

```ts
import { type Arithmetic, gradient } from 'adjointly';

function hinge<T>(A: Arithmetic<T>, [x]: readonly T[]): T {
  return A.value(x!) > 1 ? A.constant(0) : A.sub(A.constant(1), x!);
}
gradient(hinge, [0.5]).gradient; //=> [-1]
gradient(hinge, [2]).gradient; //=> [0]
gradient(hinge, [1]).gradient; //=> [-1]
```

At a branch boundary (`x = 1` above) you get the one-sided derivative of whichever branch the
comparison chose. Where the function is continuous and smooth across the boundary that is the true
derivative; where it has a kink, it is one of the one-sided derivatives, determined by the
comparison's `>` versus `>=`.

More in [docs/concepts.md](docs/concepts.md).

## API overview

Every export is documented in its TSDoc, with an example that is run as a test. By topic:

**Writing functions**

- `interface Arithmetic<T>`: `constant(x: number): T`, `value(a: T): number`, and one method per
  primitive: `neg, add, sub, mul, div, recip, square, sqrt, exp, expm1, log, log1p, pow, abs,
min, max, sin, cos, tanh, sigmoid, softplus`.
- `type ScalarFunction = <T>(A: Arithmetic<T>, xs: readonly T[]) => T`: what the conveniences
  take.
- `const plain: Arithmetic<number>`: values only.
- Composites, generic over any `Arithmetic`: `sum(A, xs)`, `dot(A, xs, ys)`,
  `logsumexp(A, xs)`, `softmax(A, xs): T[]`.

**One-call conveniences**

- `evaluate(f, at): number`
- `gradient(f, at, tape?): { value: number; gradient: number[] }`
- `jvp(f, at, direction, forward?): { value: number; derivative: number }`

**Reverse mode: `class Tape implements Arithmetic<Var>`**

- `new Tape({ capacity? })`; `size`, `capacity`
- `input(x): Var`, `inputs(xs): Var[]`, `constant(x): Var`, `value(v): number`
- `backward(output, seed = 1)`, `vjp(outputs, weights)`
- `adjoint(v): number`, `adjoints(vars, out?): Float64Array`,
  `gradient(output, inputs, out?): Float64Array`
- `clear()`, `mark(): Mark`, `rewind(mark)`

**Forward mode: `class Forward implements Arithmetic<Dual>`**

- `new Forward({ capacity? })`; `size`
- `input(x, tangent): Dual`, `constant(x): Dual`, `value(d): number`, `tangent(d): number`
- `clear()`

**Checking your own functions**

- `finiteDifference(f, at, h = DEFAULT_STEP): number[]`
- `checkGradient(f, at, { h?, rtol?, atol? })` returns
  `{ ok, value, analytic, numeric, worst: { index, analytic, numeric, error, allowed } }`

**The primitive table**

- `primitives`: each primitive's `value` and local partials (`deriv`, or `da` and `db`);
  `unaryNames`, `binaryNames`.

The full reference, recipes and pitfalls are in [docs/](docs/README.md).

### Using a tape directly

The conveniences allocate a little per call. In an inner loop, keep one `Tape` and reuse it:

```ts
import { Tape, type Var } from 'adjointly';

const tape = new Tape();
const out = new Float64Array(2);
let w: [number, number] = [0, 0];
for (let step = 0; step < 200; step++) {
  tape.clear();
  const [a, b] = tape.inputs(w) as [Var, Var];
  // loss = (a - 3)^2 + (b + 1)^2
  const loss = tape.add(
    tape.square(tape.sub(a, tape.constant(3))),
    tape.square(tape.add(b, tape.constant(1))),
  );
  tape.gradient(loss, [a, b], out);
  w = [w[0] - 0.1 * out[0]!, w[1] - 0.1 * out[1]!];
}
w; //=> ~[3, -1]
```

`mark()` and `rewind()` keep a shared prefix (parameters, precomputed features) on the tape while
discarding everything after it; `vjp` seeds several outputs at once. See
[docs/recipes.md](docs/recipes.md).

## A worked example

[examples/production-line/](examples/production-line/README.md) schedules maintenance on a
simulated production line, minute by minute over a shift. One forward and one backward pass give
the sensitivity of profit to every station-minute's maintenance decision; that gradient estimates
the effect of every candidate change to the schedule; a local search that tries the changes in
that order reaches 90% of the attainable gain in about a seventh of the evaluations random order
needs (and, near the optimum, no faster: the example says why). Forward mode gives the
sensitivity to the price. Run it with `npm run example`.

## Subgradient conventions

Where a primitive has no derivative, it uses a fixed, documented one:

| Primitive                | Where   | Derivative used                                                         |
| ------------------------ | ------- | ----------------------------------------------------------------------- |
| `abs(x)`                 | `x = 0` | `0`                                                                     |
| `max(a, b)`, `min(a, b)` | `a = b` | all to the first argument: `∂/∂a = 1`, `∂/∂b = 0`                       |
| `pow(a, b)`              | `a ≤ 0` | `∂/∂b = 0` (the real power is not differentiable in the exponent there) |
| `pow(a, b)`              | `b = 0` | `∂/∂a = 0` (so `0^0` does not give `0 · ∞`)                             |
| `sqrt(x)`                | `x = 0` | `+∞`, the true one-sided limit                                          |

Two propagation rules complete the picture: in reverse mode a node whose adjoint is exactly zero
passes nothing back, and in forward mode a term whose tangent is exactly zero is skipped. So an
infinite or `NaN` partial on a path that does not matter (a `sqrt(0)` multiplied by zero, an input
the direction does not move) never turns into `0 · ∞ = NaN`.

## Numerical stability

- `sigmoid` and `softplus` are primitives, evaluated without overflow for any argument, with
  derivatives that keep full relative precision in the tails. Build them from `exp` and `log`
  yourself and `softplus(800)` is `Infinity`.
- `expm1` and `log1p` are accurate near `0`, where `exp(x) - 1` and `log(1 + x)` lose every digit.
- `logsumexp` and `softmax` subtract the largest input (as a constant) before exponentiating. The
  shift cancels exactly in the math, so the gradient is exact, and nothing overflows.
- `square(x)` is one node and exact; prefer it to `pow(x, constant(2))`, whose exponent derivative
  needs `log(x)`.
- Derivatives are those of the floating-point program you wrote. A formula that cancels
  catastrophically in value usually does in its derivative too.

## Limitations

- Scalars only: no tensors, broadcasting or vectorised kernels. A vector is an array of `T`.
- First derivatives only: no Hessians or higher order (the tape cannot be differentiated).
- No GPU or WASM backend; this is plain JavaScript on `Float64Array`s.
- Not symbolic: the tape is a record of one evaluation, valid at one point.
- A `Var` is only checked by the type system, not at run time: using one from another tape, or
  after `clear()` or `rewind()` past it, silently reads the wrong node.
- Operations on values outside a primitive's domain produce whatever IEEE arithmetic does
  (`log(-1)` is `NaN`), in value and derivative alike.

## Performance

A tape node is five typed-array slots (value, two parent indices, two partials). The arrays double
when full, so once a tape has grown to the size of a computation and is reused with `clear()` or
`rewind()`, recording allocates nothing. The backward sweep is one tight loop over those arrays.
`npm run bench` times two workloads (a small neural network's loss over 1000 samples, and a
transcendental-heavy function of 20 000 inputs, both about 1.5 to 2 × 10⁵ nodes). On an
Apple-silicon laptop with Node 26, relative to evaluating the same function on plain numbers
(the median of several runs; individual runs vary by ±15%):

|                                | record the tape | record + backward (full gradient) | forward mode (one direction) |
| ------------------------------ | --------------- | --------------------------------- | ---------------------------- |
| network (81 inputs)            | 1.8×            | 2.5×                              | 1.6×                         |
| transcendental (20 000 inputs) | 1.7×            | 2.2×                              | 1.6×                         |

A full gradient costs about two and a half plain evaluations, whatever the number of inputs.

## For AI agents

Rules for writing code that this library differentiates correctly. They are short because each
one is absolute.

1. **Route every operation on a differentiable value through the arithmetic object.**
   `A.add(x, y)`, never `x + y`; `A.exp(x)`, never `Math.exp(x)`. `Var` and `Dual` are numbers
   underneath, so `x + y` type-checks and silently adds two _indices_. There is no runtime check.
2. **Write the function generic in `T`**, as `function f<T>(A: Arithmetic<T>, xs: readonly T[]): T`,
   and run it with `plain`, a `Tape` or a `Forward`. Do not write a separate plain version; the
   generic one evaluated with `plain` gives bit-for-bit the same value.
3. **Numbers go in through `A.constant(c)`.** Data, hyperparameters, literals: all constants. Their
   derivative is zero.
4. **Numbers come out through `A.value(x)`, and they are constants from then on.**
   `A.constant(A.value(x))` cuts the derivative: `A.mul(x, A.constant(A.value(x)))` has derivative
   `x`, not `2x`. Use `A.value` only to branch, to log, or to return a result.
5. **Branching on `A.value(x)` is fine; the gradient is that of the branch taken.** At the exact
   boundary it is the taken branch's one-sided derivative. Use `A.abs`, `A.max`, `A.min` for kinks
   rather than hand-written branches, so the documented subgradient applies.
6. **Use the stable primitives**: `A.sigmoid`, `A.softplus`, `A.log1p`, `A.expm1`, `A.square`,
   and the composites `logsumexp` and `softmax`. Never assemble them from `exp` and `log`.
7. **Keep domains explicit.** `A.log`, `A.sqrt`, `A.div`, `A.pow` with a non-positive base give
   `NaN` or `Infinity` outside their domain, in value and gradient. Clamp or reparametrise first
   (`A.softplus(raw)` for a positive parameter).
8. **In hot loops, reuse one `Tape`** and call `clear()` (or `rewind(mark)`) per evaluation; never
   construct a tape per evaluation. Pass an output `Float64Array` to `tape.gradient`.
9. **A `Var` dies with its tape's contents.** After `clear()`, or `rewind(m)` to before it was
   made, a `Var` refers to nothing (or to a newer node). Keep no `Var`s across evaluations except
   those made before a `mark()` you rewind to.
10. **Verify with `checkGradient(f, at)`** at a few random points away from kinks and branch
    boundaries, in a test, whenever you write a nontrivial function.
11. **Under `noUncheckedIndexedAccess`**, `xs[i]` and destructured `[x, y]` need `!`, and a
    variable seeded from an index needs an explicit type: `let s: T = xs[0]!`.

More: [docs/pitfalls.md](docs/pitfalls.md), and [llms.txt](llms.txt) for a compact summary.

## Design notes

- **One function, three arithmetics.** The alternatives, operator overloading through wrapper
  objects or a separate tracing API, either allocate an object per operation or make users keep
  two versions of their code. An interface generic in `T` costs one method call per operation and
  keeps a single source.
- **Branded indices.** `Var` is `number & { brand }`: no allocation, and passing a raw number where
  a variable belongs is a compile error. The brand cannot tell two tapes apart; that check would
  cost a field per variable.
- **Partials at record time.** Each node stores its local partials as it is recorded, so the
  backward sweep is two multiply-adds per node with no dispatch on the operation. A missing
  argument (both of a leaf's, the second of a unary node's) points at a sink slot with a zero
  partial, so the sweep needs no test for it; leaves are still skipped, for speed.
- **One table of primitives** (`primitives`) holds every value formula and every partial. `plain`
  uses its functions directly; `Tape` and `Forward` have one method per primitive, each calling its
  own table entry. A single generic recording helper taking the table entry as an argument was
  about 2× slower, because its call sites see every primitive and V8 cannot inline them.
- **Zero adjoints and zero tangents propagate nothing**, rather than multiplying through. It costs
  one comparison per node and removes the `0 · ∞ = NaN` failures of unused singular paths.
- **The composites need no rules of their own.** `logsumexp` and `softmax` are recorded as ordinary
  primitives; the max-shift is a constant, which is exact because the shift cancels.
- **`gradient()` returns `number[]`, `Tape.gradient()` a `Float64Array`.** The convenience is for
  readability and serialisation; the tape method writes into a buffer you can reuse.
- **`Forward` has `clear()` but no `mark()`/`rewind()`.** Forward mode keeps nothing that a later
  evaluation could reuse except inputs, which are cheap to recreate.

## Development

See [AGENTS.md](AGENTS.md) for the layout, the invariants and the gates. In short: `npm run check`
runs formatting, lint, type checking, tests with a 100% coverage gate, the build, publint, attw and
the API report check; `npm run mutate` runs mutation testing; `npm run bench` times the workloads.
Changes that affect users get a changeset (`npx changeset`).
