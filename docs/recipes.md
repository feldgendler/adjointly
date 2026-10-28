# Recipes

## Gradient descent in an inner loop

Keep one tape for the whole loop and clear it per step; pass a buffer for the gradient. After the
first step the tape has grown to the size of the computation and nothing more is allocated.

```ts
import { type Arithmetic, Tape } from 'adjointly';

// Least squares: fit y = a x + b to three points.
const xs = [0, 1, 2];
const ys = [1, 3, 5];
function loss<T>(A: Arithmetic<T>, a: T, b: T): T {
  let s = A.constant(0);
  for (let i = 0; i < xs.length; i++) {
    const r = A.sub(A.add(A.mul(a, A.constant(xs[i]!)), b), A.constant(ys[i]!));
    s = A.add(s, A.square(r));
  }
  return s;
}

const tape = new Tape();
const grad = new Float64Array(2);
let a = 0;
let b = 0;
for (let step = 0; step < 2000; step++) {
  tape.clear();
  const av = tape.input(a);
  const bv = tape.input(b);
  tape.gradient(loss(tape, av, bv), [av, bv], grad);
  a -= 0.05 * grad[0]!;
  b -= 0.05 * grad[1]!;
}
[a, b]; //=> ~[2, 1]
```

## A shared prefix with `mark` and `rewind`

When many evaluations share their first part (parameters, features computed from them), record it
once, mark the tape, and rewind to the mark for each evaluation. Vars recorded before the mark stay
valid.

```ts
import { Tape } from 'adjointly';

const t = new Tape();
const w = t.input(2);
const w2 = t.square(w); // shared: recorded once
const m = t.mark();
let total = 0;
for (const x of [1, 2, 3]) {
  t.rewind(m);
  // d/dw of w^2 * x = 2 w x
  t.backward(t.mul(w2, t.constant(x)));
  total += t.adjoint(w);
}
total; //=> 2 * 2 * (1 + 2 + 3)
```

## Several outputs: one sweep each, or one weighted sweep

```ts
import { Tape } from 'adjointly';

const t = new Tape();
const [x, y] = t.inputs([1, 2]);
const f = t.mul(x!, y!);
const g = t.add(x!, t.square(y!));

t.backward(f);
t.adjoints([x!, y!]); //=> [2, 1]
t.backward(g);
t.adjoints([x!, y!]); //=> [1, 4]
t.vjp([f, g], [10, 1]); // gradient of 10 f + g
t.adjoints([x!, y!]); //=> [21, 14]
```

## A full Jacobian

Reverse mode gives a row per sweep; forward mode a column per pass. Use whichever needs fewer.

```ts
import { type Arithmetic, Forward, Tape } from 'adjointly';

// F(x, y) = (x y, x + y, sin x)
function F<T>(A: Arithmetic<T>, x: T, y: T): T[] {
  return [A.mul(x, y), A.add(x, y), A.sin(x)];
}

// Rows: one recording, one sweep per output.
const t = new Tape();
const x = t.input(2);
const y = t.input(3);
const outs = F(t, x, y);
const rows = outs.map((o) => Array.from(t.gradient(o, [x, y])));
rows; //=> [[3, 2], [1, 1], [cos(2), 0]]

// Columns: one forward pass per input.
const cols = [0, 1].map((j) => {
  const fw = new Forward();
  const ys = F(fw, fw.input(2, j === 0 ? 1 : 0), fw.input(3, j === 1 ? 1 : 0));
  return ys.map((v) => fw.tangent(v));
});
cols; //=> [[3, 1, cos(2)], [2, 1, 0]]
```

## Constrained parameters

Optimise an unconstrained number and map it into the domain with a smooth primitive: `softplus`
for positive, `sigmoid` for a probability, `exp` for a scale.

```ts
import { type Arithmetic, gradient } from 'adjointly';

// A rate that must stay positive, parametrised by an unconstrained raw value.
function negLogLikelihood<T>(A: Arithmetic<T>, [raw]: readonly T[]): T {
  const rate = A.softplus(raw!);
  const waits = [0.5, 1.5, 1.0];
  // exponential distribution: -sum(log(rate) - rate * w)
  let s = A.constant(0);
  for (const w of waits) s = A.sub(s, A.sub(A.log(rate), A.mul(rate, A.constant(w))));
  return s;
}
// At the optimum rate = 1 / mean(waits) = 1, i.e. raw = log(e - 1), the gradient vanishes.
gradient(negLogLikelihood, [Math.log(Math.E - 1)]).gradient; //=> ~[0]
```

## Test your own gradient

```ts
import { checkGradient, logsumexp, type ScalarFunction } from 'adjointly';

const f: ScalarFunction = (A, xs) =>
  A.add(logsumexp(A, xs), A.mul(A.tanh(xs[0]!), A.sigmoid(xs[1]!)));

for (const at of [
  [0.1, -0.7, 2],
  [-3, 0.4, 0.5],
]) {
  const r = checkGradient(f, at);
  if (!r.ok) throw new Error(`component ${String(r.worst.index)} is off`);
}
checkGradient(f, [1, 2, 3]).ok; //=> true
```

## The worked example

[examples/production-line/](https://github.com/feldgendler/adjointly/tree/main/examples/production-line) (in the repository, not the npm package) puts these together: a
minute-by-minute simulation written once, its gradient with respect to 1 440 decisions from one
tape pass, first-order estimates ranking discrete moves, and forward mode for a parameter.
