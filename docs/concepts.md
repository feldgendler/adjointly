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
independent of the number of inputs; the memory is six numbers per node, counting its adjoint
(40 bytes: four 8-byte floats and two 4-byte indices).

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
function newtonSqrt<T>(A: Arithmetic<T>, [a]: readonly [T]): T {
  let x = A.constant(1);
  for (;;) {
    const next = A.mul(A.constant(0.5), A.add(x, A.div(a, x)));
    if (Math.abs(A.value(next) - A.value(x)) < 1e-15) return next;
    x = next;
  }
}
gradient(newtonSqrt, [4]); //=> { value: 2, gradient: [0.25] }
```

What the tape cannot see is a dependence that goes through `A.value`: a number read out and fed
back in with `A.constant` is a constant. That is the right behaviour for a branch condition or a
step size, and the wrong one anywhere a derivative should flow (see [pitfalls.md](pitfalls.md)).

## The zero rule

Every term of the chain rule is a product of two factors: a local partial, and the adjoint flowing
back into the node (reverse mode) or the tangent flowing forward out of an argument (forward mode).
**If either factor is exactly 0, the term is 0**, even when the other factor is infinite or `NaN`.
Both modes apply the rule. Without it, `0 * Infinity` would turn an infinite partial on a path that
does not matter into `NaN`:

```ts
import { type Arithmetic, gradient, jvp } from 'adjointly';

// sqrt at 0 has an infinite derivative, but here it is multiplied by 0.
function f<T>(A: Arithmetic<T>, [x, y]: readonly [T, T]): T {
  return A.add(A.mul(A.constant(0), A.sqrt(x)), y);
}
gradient(f, [0, 1]).gradient; //=> [0, 1]
jvp(f, [0, 1], [1, 0]).derivative; //=> 0
jvp(f, [0, 1], [0, 1]).derivative; //=> 1

// The zero comes before the singular point: 0 * x is 0 for every x, so sqrt(0 * x) is constant.
function g<T>(A: Arithmetic<T>, [x]: readonly [T]): T {
  return A.sqrt(A.mul(A.constant(0), x));
}
gradient(g, [1]).gradient; //=> [0]
jvp(g, [1], [1]).derivative; //=> 0
```

The two modes multiply the same partials, grouped differently: forward mode from the inputs
towards the output, reverse mode from the output back. The rule makes them agree, but two things
can still tell the groupings apart:

- **Cancellation.** The rule looks at factors, not sums. In `sqrt(x - x)` the tangent of `x - x`
  is `1 - 1 = 0` before it meets the infinite partial of `sqrt` at 0, so forward mode gives 0;
  reverse mode multiplies the infinite partial into each of the two terms first and adds
  `∞ - ∞ = NaN`.
- **Underflow and overflow.** A product of tiny (or huge) finite partials can round to 0 (or
  overflow to `Infinity`) in one grouping and not in the other, and a 0 produced that way then
  annihilates whatever it meets.

Where neither happens, the modes agree, `NaN` for `NaN`; a property test checks this on random
expressions at 0, -0, ±Infinity, `NaN` and subnormal inputs.
