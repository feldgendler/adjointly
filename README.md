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

> **Status:** not yet published; the first release will be 0.1.0, and the API may still change
> before 1.0. **License:** [MIT](LICENSE).

## Install

```sh
npm install adjointly
```

The package is not on npm yet; until it is, build it from this repository (`npm ci && npm run
build`) and depend on the checkout.

## Quick start

```ts
import { type Arithmetic, evaluate, gradient, jvp } from 'adjointly';

// f(x, y) = sin(x) * e^y, written once against the arithmetic interface
function f<T>(A: Arithmetic<T>, [x, y]: readonly [T, T]): T {
  return A.mul(A.sin(x), A.exp(y));
}

evaluate(f, [1, 0]); //=> sin(1)
gradient(f, [1, 0]); //=> { value: sin(1), gradient: [cos(1), sin(1)] }
jvp(f, [1, 0], [1, 1]); //=> { value: sin(1), derivative: cos(1) + sin(1) }
```

Typing the inputs as a tuple, `readonly [T, T]`, gives each destructured input the type `T`, even
under `noUncheckedIndexedAccess`; a point written as a literal (`[1, 0]`) is checked against it.

## Concepts

**The arithmetic interface.** A differentiable function takes an `Arithmetic<T>` (here `A`) and
values of an opaque type `T`, and does every operation through `A`: `A.add(x, y)`, `A.exp(x)`.
`T` is `number` when the function is evaluated plainly, and an index into storage when it is being
differentiated. Numbers enter through `A.constant(c)`, or as the plain-number operand of `addc`,
`mulc` and the like; values leave through `A.value(x)`.

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

function hinge<T>(A: Arithmetic<T>, [x]: readonly [T]): T {
  return A.value(x) > 1 ? A.constant(0) : A.csub(1, x);
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
min, max, sin, cos, tanh, sigmoid, softplus`; and, for a plain-number operand `c`,
  `addc(a, c), subc(a, c), csub(c, a), mulc(a, c), divc(a, c), cdiv(c, a), powc(a, c)`: the
  binary primitive with a constant, in one tape node instead of two.
- `type ScalarFunction<X = readonly number[]> = <T>(A: Arithmetic<T>, xs: Inputs<T, X>) => T`:
  what the conveniences take. `Inputs<T, X>` is one `T` per number of the point `X`: a tuple for a
  literal point such as `[1, 0]`, `readonly T[]` for a `number[]`.
- `const plain: Arithmetic<number>`: values only.
- Composites, generic over any `Arithmetic`: `sum(A, xs)`, `dot(A, xs, ys)`,
  `logsumexp(A, xs)`, `softmax(A, xs): T[]`.

**One-call conveniences**

- `evaluate(f, at): number`
- `gradient(f, at, tape?): { value: number; gradient: number[] }`
- `jvp(f, at, direction, forward?): { value: number; derivative: number }`

**Reverse mode: `class Tape implements Arithmetic<Var>`**

- `new Tape({ capacity?, debug? })`; `size`, `capacity`
- `input(x): Var`, `inputs(xs): Var[]` (a tuple for a literal), `constant(x): Var`, `value(v): number`
- `backward(output, seed = 1)`, `vjp(outputs, weights)`
- `adjoint(v): number`, `adjoints(vars, out?): Float64Array`,
  `gradient(output, inputs, out?): Float64Array`
- `clear()`, `mark(): Mark`, `rewind(mark)`

**Forward mode: `class Forward implements Arithmetic<Dual>`**

- `new Forward({ capacity?, debug? })`; `size`
- `input(x, tangent): Dual`, `constant(x): Dual`, `value(d): number`, `tangent(d): number`
- `clear()`

**Checking your own functions**

- `finiteDifference(f, at, h = DEFAULT_STEP): number[]`
- `checkGradient(f, at, { h?, rtol?, atol? })` returns
  `{ ok, value, analytic, numeric, worst: { index, analytic, numeric, error, allowed, roundoff } }`

**The primitive table**

- `primitives`: each primitive's `value` and local partials (`deriv`, or `da` and `db`);
  `unaryNames`, `binaryNames`.

The full reference, recipes and pitfalls are in [docs/](docs/README.md).

### Using a tape directly

The conveniences allocate a little per call. In an inner loop, keep one `Tape` and reuse it:

```ts
import { Tape } from 'adjointly';

const tape = new Tape();
const out = new Float64Array(2);
let w: [number, number] = [0, 0];
for (let step = 0; step < 200; step++) {
  tape.clear();
  const [a, b] = tape.inputs(w);
  // loss = (a - 3)^2 + (b + 1)^2
  const loss = tape.add(tape.square(tape.subc(a, 3)), tape.square(tape.addc(b, 1)));
  tape.gradient(loss, [a, b], out);
  w = [w[0] - 0.1 * out[0]!, w[1] - 0.1 * out[1]!];
}
w; //=> ~[3, -1]
```

`mark()` and `rewind()` keep a shared prefix (parameters, precomputed features) on the tape while
discarding everything after it; `vjp` seeds several outputs at once. See
[docs/recipes.md](docs/recipes.md).

## A worked example

[examples/production-line/](https://github.com/feldgendler/adjointly/tree/main/examples/production-line) schedules maintenance on a simulated production line, minute by minute over a shift. One
forward and one backward pass give the sensitivity of profit to every station-minute's
maintenance decision, and that gradient estimates the effect of every candidate change to the
schedule. A local search that tries the changes in that order gets most of the way faster than one
that tries them at random (three quarters of the best gain in about a third of the work, gradient
passes counted),
but is slower to finish and costlier overall: near an optimum the first-order estimate no longer
ranks the moves. Forward mode gives the sensitivity to the price. Run it with `npm run example`.

## Subgradient conventions

Where a primitive has no derivative, it uses a fixed, documented one:

| Primitive                | Where               | Derivative used                                                         |
| ------------------------ | ------------------- | ----------------------------------------------------------------------- |
| `abs(x)`                 | `x = 0`             | `0`                                                                     |
| `max(a, b)`, `min(a, b)` | `a = b`             | all to the first argument: `∂/∂a = 1`, `∂/∂b = 0`                       |
| `pow(a, b)`              | `a ≤ 0`             | `∂/∂b = 0` (the real power is not differentiable in the exponent there) |
| `pow(a, b)`              | `b = 0`             | `∂/∂a = 0` (so `0^0` does not give `0 · ∞`)                             |
| `sqrt(x)`                | `x = 0` or `x = -0` | `+∞`, the true one-sided limit                                          |
| `max(a, b)`, `min(a, b)` | `a` or `b` is `NaN` | `NaN` for both, like `abs(NaN)`                                         |

One rule completes the picture, the **zero rule**: every term of the chain rule is a local partial
times an incoming adjoint (reverse mode) or tangent (forward mode), and if either factor is exactly
0 the term is 0, even when the other is infinite or `NaN`. So an infinite partial on a path that
does not matter (a `sqrt(0)` multiplied by zero, the untaken side of a `max`, an input the
direction does not move) never turns into `0 · ∞ = NaN`, and both modes give the same answer.
The rule looks at factors, not sums, so it cannot see a cancellation: `sqrt(x - x)` has derivative
0 in forward mode and `NaN` in reverse mode, which multiplies the infinite partial into each term
before they cancel. [docs/concepts.md](docs/concepts.md#the-zero-rule) has the details.

## Numerical stability

- `sigmoid` and `softplus` are primitives, evaluated without overflow for any argument, with
  derivatives that keep full relative precision in the tails. Build them from `exp` and `log`
  yourself and `softplus(800)` is `Infinity`.
- `tanh`'s derivative is computed from `exp(-2|x|)`, not as `1 - tanh(x)^2`, which cancels to 0
  in the tails.
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
  after `clear()` or `rewind()` past it, silently reads the wrong node. `new Tape({ debug: true })`
  checks every operand and throws instead, at some cost in speed.
- A `Var` is a number at run time, so `typeof x === 'number'` cannot tell it from a number; see
  rule 1 of "For AI agents" below.
- Operations on values outside a primitive's domain produce whatever IEEE arithmetic does
  (`log(-1)` is `NaN`), in value and derivative alike.

## Performance

A tape node is six typed-array slots (value, two parent indices, two partials and its adjoint:
40 bytes). The arrays double when full, so once a tape has grown to the size of a computation and
is reused with `clear()` or `rewind()`, recording allocates nothing. The backward sweep is one tight loop over those arrays.
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

1. **Never tell a variable from a plain number by `typeof x === 'number'` or by its value.** A `Var`
   is a number at run time and passes that test, so code that lifts "numbers" into constants that
   way turns every variable into a constant, and every gradient is silently 0. Keep variables and
   numbers in separate places, and give numbers to `addc`, `mulc` and the other plain-number
   operations. When gradients look wrong, run once with `new Tape({ debug: true })`, which throws at
   the first operand that is not a live variable of that tape, and at a variable where a number
   belongs.
2. **Route every operation on a differentiable value through the arithmetic object.**
   `A.add(x, y)`, never `x + y`; `A.exp(x)`, never `Math.exp(x)`. `Var` and `Dual` are numbers
   underneath, so `x + y` type-checks and silently adds two _indices_. There is no runtime check.
3. **Write the function generic in `T`**, as `function f<T>(A: Arithmetic<T>, xs: readonly T[]): T`,
   and run it with `plain`, a `Tape` or a `Forward`. Do not write a separate plain version; the
   generic one evaluated with `plain` gives bit-for-bit the same value.
4. **Numbers go in through `A.constant(c)`, or as the number operand of `addc`, `subc`, `csub`,
   `mulc`, `divc`, `cdiv`, `powc`.** Data, hyperparameters, literals: all constants, with derivative
   zero. Prefer the number-operand operations: `A.mulc(x, 2)` is one tape node,
   `A.mul(x, A.constant(2))` two, with the same value and derivative.
5. **Numbers come out through `A.value(x)`, and they are constants from then on.**
   `A.constant(A.value(x))` cuts the derivative: `A.mul(x, A.constant(A.value(x)))` has derivative
   `x`, not `2x`. Use `A.value` only to branch, to log, or to return a result.
6. **Branching on `A.value(x)` is fine; the gradient is that of the branch taken.** At the exact
   boundary it is the taken branch's one-sided derivative. Use `A.abs`, `A.max`, `A.min` for kinks
   rather than hand-written branches, so the documented subgradient applies.
7. **Use the stable primitives**: `A.sigmoid`, `A.softplus`, `A.log1p`, `A.expm1`, `A.square`,
   and the composites `logsumexp` and `softmax`. Never assemble them from `exp` and `log`.
8. **Keep domains explicit.** `A.log`, `A.sqrt`, `A.div`, `A.pow` with a non-positive base give
   `NaN` or `Infinity` outside their domain, in value and gradient. Clamp or reparametrise first
   (`A.softplus(raw)` for a positive parameter).
9. **In hot loops, reuse one `Tape`** and call `clear()` (or `rewind(mark)`) per evaluation; never
   construct a tape per evaluation. Pass an output `Float64Array` to `tape.gradient`.
10. **A `Var` dies with its tape's contents.** After `clear()`, or `rewind(m)` to before it was
    made, a `Var` refers to nothing (or to a newer node). Keep no `Var`s across evaluations except
    those made before a `mark()` you rewind to.
11. **Verify with `checkGradient(f, at)`** at a few random points away from kinks and branch
    boundaries, in a test, whenever you write a nontrivial function.
12. **Give a fixed number of inputs a tuple type**, `[x, y]: readonly [T, T]` (or
    `ScalarFunction<[number, number]>`), so they need no `!` under `noUncheckedIndexedAccess`; with
    a `readonly T[]` of any length, index with `!` and annotate an accumulator seeded from it:
    `let s: T = xs[0]!`.

More: [docs/pitfalls.md](docs/pitfalls.md), and [llms.txt](llms.txt) for a compact summary.

## Design notes

- **Plain-number operands: `addc`, `subc`, `csub`, `mulc`, `divc`, `cdiv`, `powc`.** In real code
  most operands of a recurrence are fixed parameters and data, and spelling each with
  `constant()` doubles the tape. The `c` marks the side the constant is on, and the arguments are
  in the order of the expression: `subc(a, c)` is `a - c`, `csub(c, a)` is `c - a`. Short names
  rather than `scale` or `shift`, because subtraction and division need both orders and a power
  needs one, and a uniform suffix covers them all. Each is the binary primitive with one operand
  fixed, computed from that primitive's table entry, so its value and derivative are bit for bit
  those of the constant spelling. There is no `minc`/`maxc`: use `max(x, constant(c))`.

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
- **The zero rule** (a zero partial or a zero adjoint or tangent makes the term zero) costs a
  comparison per term and removes the `0 · ∞ = NaN` failures of unused singular paths, in both
  modes alike, so they agree wherever their different groupings of the chain rule cannot matter.
- **The composites need no rules of their own.** `logsumexp` and `softmax` are recorded as ordinary
  primitives; the max-shift is a constant, which is exact because the shift cancels.
- **`gradient()` returns `number[]`, `Tape.gradient()` a `Float64Array`.** The convenience is for
  readability and serialisation; the tape method writes into a buffer you can reuse.
- **`Forward` has `clear()` but no `mark()`/`rewind()`.** Forward mode keeps nothing that a later
  evaluation could reuse except inputs, which are cheap to recreate.

## Development

See [AGENTS.md](AGENTS.md) for the layout, the invariants and the gates. In short: `npm run check`
runs formatting, lint, type checking, tests with a 100% coverage gate, the build, publint, attw and
the API report check; `npm run mutate` runs mutation testing; `npm run bench` times the workloads,
and `npm run bench:ab` compares them with another checkout, as CI does for every pull request.
Changes that affect users get a changeset (`npx changeset`).
