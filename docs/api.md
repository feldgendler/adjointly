# API

Everything is exported from the package root, `adjointly`. The TSDoc on each export is the
authoritative contract; this page groups them by topic. The committed API report,
`etc/adjointly.api.md` in the repository, lists every signature.

## Writing a differentiable function

### `interface Arithmetic<T>`

| member                                                                                                 | contract                                                          |
| ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `constant(x: number): T`                                                                               | a number as a constant: zero derivative. Appends a node on a tape |
| `value(a: T): number`                                                                                  | the current value; a plain number with no derivative attached     |
| `neg, recip, square, sqrt, exp, expm1, log, log1p, abs, sin, cos, tanh, sigmoid, softplus` `(a: T): T` | one-argument primitives                                           |
| `add, sub, mul, div, pow, min, max` `(a: T, b: T): T`                                                  | two-argument primitives                                           |

Every operation on a `T` must go through these methods.

### `type ScalarFunction = <T>(A: Arithmetic<T>, xs: readonly T[]) => T`

The shape the conveniences accept: a function of a list of inputs, generic in `T`.

### `const plain: Arithmetic<number>`

Values only, nothing recorded.

```ts
import { plain } from 'adjointly';

plain.softplus(plain.constant(-800)); //=> 0
plain.softplus(800); //=> 800
```

### Composites: `sum`, `dot`, `logsumexp`, `softmax`

Generic over any `Arithmetic`, built from primitives, so they need no derivative rules.

| function              | contract                                                                       |
| --------------------- | ------------------------------------------------------------------------------ |
| `sum(A, xs): T`       | left-to-right sum; `constant(0)` for an empty list                             |
| `dot(A, xs, ys): T`   | `sum_i xs[i] * ys[i]`; throws `RangeError` on different lengths                |
| `logsumexp(A, xs): T` | `log(sum_i exp(xs[i]))`, shifted by the largest value; throws on an empty list |
| `softmax(A, xs): T[]` | `exp(xs[i]) / sum_j exp(xs[j])`, shifted likewise; throws on an empty list     |

```ts
import { gradient, logsumexp, softmax, plain } from 'adjointly';

logsumexp(plain, [1000, 1000]); //=> 1000 + log(2)
softmax(plain, [0, Math.log(3)]); //=> [0.25, 0.75]
// The gradient of logsumexp is the softmax.
gradient((A, xs) => logsumexp(A, xs), [0, Math.log(3)]).gradient; //=> [0.25, 0.75]
```

## Conveniences

| function                          | returns                                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `evaluate(f, at)`                 | `f(at)` on plain numbers                                                                                 |
| `gradient(f, at, tape?)`          | `{ value, gradient: number[] }` by reverse mode; reuses `tape` if given                                  |
| `jvp(f, at, direction, forward?)` | `{ value, derivative }`, the derivative along `direction`, by forward mode; throws if the lengths differ |

```ts
import { evaluate, gradient, jvp, type ScalarFunction } from 'adjointly';

const f: ScalarFunction = (A, [x, y]) => A.div(x!, y!);
evaluate(f, [3, 4]); //=> 0.75
gradient(f, [3, 4]); //=> { value: 0.75, gradient: [0.25, -3 / 16] }
jvp(f, [3, 4], [1, 1]); //=> { value: 0.75, derivative: 0.25 - 3 / 16 }
```

## Reverse mode: `class Tape implements Arithmetic<Var>`

| member                           | contract                                                                                                   |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `new Tape({ capacity? })`        | an empty tape; capacity in nodes, default 1024, doubles when full                                          |
| `size`, `capacity`               | nodes recorded; nodes it can hold before growing                                                           |
| `input(x)`, `inputs(xs)`         | leaves to differentiate with respect to                                                                    |
| `constant(x)`                    | a leaf too; the name only says you do not care for its adjoint                                             |
| `value(v)`                       | the value of a node                                                                                        |
| `backward(output, seed = 1)`     | one sweep: afterwards `adjoint(v)` is `seed * d(output)/d(v)`. Throws if `output` is not on the tape       |
| `vjp(outputs, weights)`          | one sweep seeded with every `weights[k]` at `outputs[k]`. Throws on mismatched lengths or a foreign output |
| `adjoint(v)`                     | the adjoint from the last sweep; zero for nodes the output does not depend on                              |
| `adjoints(vars, out?)`           | the adjoints of `vars` into a `Float64Array`                                                               |
| `gradient(output, inputs, out?)` | `backward(output)` then `adjoints(inputs, out)`                                                            |
| `clear()`                        | forget every node, keep the storage                                                                        |
| `mark()`, `rewind(mark)`         | truncate back to a checkpoint, keeping the nodes before it. `rewind` throws if the mark is beyond the end  |

Sweeps leave the tape unchanged: sweep it as often as you like, from different outputs.

### `type Var`, `type Mark`

`Var` is `number & { brand }`: an index into its tape. A raw number is not assignable to it, and
neither is a `Dual`. `Mark` is a branded tape length.

## Forward mode: `class Forward implements Arithmetic<Dual>`

| member                       | contract                                                 |
| ---------------------------- | -------------------------------------------------------- |
| `new Forward({ capacity? })` | an empty context                                         |
| `input(x, tangent)`          | an input with its component of the direction             |
| `constant(x)`                | tangent 0                                                |
| `value(d)`, `tangent(d)`     | the value, and the derivative along the seeded direction |
| `size`, `clear()`            | values created; forget them                              |

## Checking: `finiteDifference`, `checkGradient`, `DEFAULT_STEP`

| function                                                | contract                                                                                                                                                     |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `finiteDifference(f, at, h = DEFAULT_STEP)`             | central differences with step `h * max(1, abs(at[i]))`; two evaluations per input                                                                            |
| `checkGradient(f, at, { h, rtol = 1e-5, atol = 1e-8 })` | reverse mode against central differences; component `i` passes when `abs(analytic - numeric) <= atol + rtol * max(abs(analytic), abs(numeric))`; `NaN` fails |

`checkGradient` returns `{ ok, value, analytic, numeric, worst }`, where `worst` is the component
furthest outside its allowance, with `index`, `analytic`, `numeric`, `error` and `allowed`.

```ts
import { checkGradient, type ScalarFunction } from 'adjointly';

// A gradient bug: reading a value out and back in as a constant.
const buggy: ScalarFunction = (A, [x]) => A.mul(x!, A.constant(A.value(x!)));
const r = checkGradient(buggy, [3]);
r.ok; //=> false
r.worst.analytic; //=> 3
r.worst.numeric; //=> ~6
```

## The primitive table: `primitives`, `unaryNames`, `binaryNames`

`primitives[name]` is a frozen `UnaryPrimitive` (`value(x)`, `deriv(x, y)`) or `BinaryPrimitive`
(`value(a, b)`, `da(a, b, y)`, `db(a, b, y)`), where `y` is the operation's own result. It is the
single source of every value and partial derivative in the library.

```ts
import { binaryNames, primitives, unaryNames } from 'adjointly';

primitives.pow.db(0, 2, 0); //=> 0
primitives.abs.deriv(0, 0); //=> 0
unaryNames.length + binaryNames.length; //=> 21
```
