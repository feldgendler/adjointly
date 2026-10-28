# Concepts

## Functions written once

A function to differentiate is written against `Arithmetic<T>`, generic in the value type `T`:

```ts
import { type Arithmetic, Forward, plain, Tape } from 'adjointly';

function f<T>(A: Arithmetic<T>, x: T, y: T): T {
  return A.add(A.mul(x, y), A.sin(x)); // x*y + sin(x)
}

f(plain, 2, 3); //=> 6 + sin(2)

const t = new Tape();
const x = t.input(2);
const y = t.input(3);
t.backward(f(t, x, y));
t.adjoint(x); //=> 3 + cos(2)

const F = new Forward();
const out = f(F, F.input(2, 0), F.input(3, 1)); // direction: along y
F.tangent(out); //=> 2
```

The three implementations compute every value with the same functions from the `primitives`
table, so `f(plain, ...)`, the tape's value and the forward value are bit-for-bit identical. The
tests hold this for random expressions over every primitive.

## The tape

A `Tape` appends a node for every operation, `constant` and `input` included. Node `i` stores:

| array      | holds                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------- |
| `val`      | the value                                                                                   |
| `pa`, `pb` | the indices of the one or two arguments; a missing one is slot 0, a sink that is never read |
| `da`, `db` | the local partial derivatives with respect to them, computed when the node was recorded     |

A `Var` is just that index, branded so the type checker can tell it from a plain number. Because
the partials are computed during recording, the backward sweep does not need to know which
operation a node was:

```text
adj[0..output] = 0; adj[output] = seed
for i from output down to 1:
    if adj[i] == 0 or i is a leaf: continue
    adj[pa[i]] += da[i] * adj[i]
    adj[pb[i]] += db[i] * adj[i]     # for a unary node pb[i] is the sink and db[i] is 0
```

After the sweep `adj[v]` is d(output)/d(v) for every node `v` the output depends on. One sweep
gives the derivative with respect to every input at once, which is why reverse mode is the right
tool for a scalar loss of many parameters. The cost is a small constant times one evaluation,
independent of the number of inputs; the memory is five numbers per node.

`vjp(outputs, weights)` seeds several outputs in one sweep, giving the gradient of
`sum_k weights[k] * outputs[k]`; `backward` is the one-output case.

## Forward mode

`Forward` keeps a value and a tangent per slot. Inputs are seeded with the components of a
direction `v`; each operation computes its tangent as `da * tangent(a) + db * tangent(b)` from the
same partials. One evaluation gives `grad f . v` for every output. It needs no tape and no second
pass, so it is the better tool when there are few inputs, or only one direction of interest, and
possibly many outputs.

| inputs | outputs | what you want               | cheapest                     |
| ------ | ------- | --------------------------- | ---------------------------- |
| many   | one     | the full gradient           | one reverse sweep            |
| few    | many    | the Jacobian                | one forward pass per input   |
| many   | many    | one Jacobian-vector product | one forward pass             |
| many   | many    | one vector-Jacobian product | one reverse sweep with `vjp` |

## Control flow

The tape records what ran. Branches on `A.value(x)`, loops whose trip count depends on values,
early returns, recursion: all are fine, and the derivative is that of the path taken at this point.
At a point exactly on a branch boundary the derivative is the taken branch's one-sided derivative.

```ts
import { type Arithmetic, gradient } from 'adjointly';

// Newton's method for sqrt(a), iterated until it converges. Differentiating through the
// iterations gives the derivative of the converged result: 1 / (2 sqrt(a)).
function newtonSqrt<T>(A: Arithmetic<T>, [a]: readonly T[]): T {
  let x = A.constant(1);
  for (;;) {
    const next = A.mul(A.constant(0.5), A.add(x, A.div(a!, x)));
    if (Math.abs(A.value(next) - A.value(x)) < 1e-15) return next;
    x = next;
  }
}
gradient(newtonSqrt, [4]); //=> { value: 2, gradient: [0.25] }
```

What the tape cannot see is a dependence that goes through `A.value`: a number read out and fed
back in with `A.constant` is a constant. That is the right behaviour for a branch condition or a
step size, and the wrong one anywhere a derivative should flow (see [pitfalls.md](pitfalls.md)).

## Zero means nothing flows

A node whose adjoint is exactly zero passes nothing to its arguments, and in forward mode a term
whose tangent is exactly zero is skipped. Without that rule, `0 * Infinity` would turn an infinite
partial on an unused path into `NaN`:

```ts
import { type Arithmetic, gradient, jvp } from 'adjointly';

// sqrt at 0 has an infinite derivative, but here it is multiplied by 0.
function f<T>(A: Arithmetic<T>, [x, y]: readonly T[]): T {
  return A.add(A.mul(A.constant(0), A.sqrt(x!)), y!);
}
gradient(f, [0, 1]).gradient; //=> [0, 1]
jvp(f, [0, 1], [0, 1]).derivative; //=> 1
```
