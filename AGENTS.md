# Working on adjointly

Instructions for agents (and people) changing this library. Users of the library want
[README.md](README.md) and [docs/](docs/README.md) instead.

## Layout

| Path                        | What it is                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `src/primitives.ts`         | **The** table of primitives: every value formula and every local partial derivative       |
| `src/arithmetic.ts`         | `Arithmetic<T>`, `ScalarFunction`, and `plain` (the table's value functions)              |
| `src/tape.ts`               | `Tape`, reverse mode: typed-array storage, recording, the backward sweep, mark/rewind     |
| `src/forward.ts`            | `Forward`, forward mode: a value and a tangent per slot                                   |
| `src/composites.ts`         | `sum`, `dot`, `logsumexp`, `softmax`, generic over any `Arithmetic`                       |
| `src/convenience.ts`        | `evaluate`, `gradient`, `jvp`                                                             |
| `src/check.ts`              | `finiteDifference`, `checkGradient`                                                       |
| `src/index.ts`              | the public API: everything exported from the package root                                 |
| `test/`                     | Vitest suites; `test/support/` holds helpers, including the doc-example extractor         |
| `bench/`                    | tinybench workloads (`workloads.ts`), the runner and comparer (`run.ts`), `baseline.json` |
| `examples/production-line/` | the worked example; run by `npm run example`, tested by `test/example.test.ts`            |
| `docs/`, `llms.txt`         | user documentation, shipped in the package                                                |
| `etc/adjointly.api.md`      | API Extractor's report of the public API, committed                                       |
| `.changeset/`               | pending changesets                                                                        |

## Invariants

These are what the library promises; every change keeps them.

1. **One table of primitives.** A primitive's value and partials are written once, in
   `src/primitives.ts`. `plain` uses the table's functions directly; each `Tape` and `Forward`
   method calls its own table entry. No formula is ever written a second time anywhere.
2. **Values are bit-for-bit identical** under `plain`, `Tape` and `Forward`. This follows from (1)
   and from composites being generic code that runs the same operations in the same order under
   every arithmetic. Tested in `test/primitives.test.ts` and, for random expressions, in
   `test/properties.test.ts`.
3. **No allocation per operation.** Recording writes into typed arrays that double when full;
   `Var` and `Dual` are branded integers. Nothing on the per-operation path creates an object,
   array or closure.
4. **Monomorphic hot paths.** Each `Tape`/`Forward` method is written out separately and calls its
   own table entry. A shared helper that takes the table entry as a parameter looks tidier and is
   about 2x slower (its call sites see every primitive, so V8 cannot inline them). Do not
   "refactor" it back.
5. **The zero rule**: a chain-rule term is a local partial times an incoming adjoint (reverse) or
   tangent (forward), and if either factor is exactly 0 the term is 0, even when the other is
   infinite or `NaN`. Both modes apply it (`sweep` in tape.ts, `times` in forward.ts), so they
   agree except where cancellation or under/overflow distinguishes their groupings
   (`test/zero-rule.test.ts`, docs/concepts.md).
6. **Subgradient conventions are part of the contract** (table in `src/primitives.ts` and the
   README). Changing one is a breaking change.
7. **No runtime dependencies.** `dependencies` stays empty.

## Adding a primitive, end to end

1. `src/primitives.ts`: add the entry with `unary(...)` or `binary(...)`, its TSDoc, and its name to
   `unaryNames` or `binaryNames`. If it has a point without a derivative, add the convention to the
   table in the `primitives` TSDoc.
2. `src/arithmetic.ts`: add the method to `Arithmetic<T>` (with TSDoc) and to `plain`.
3. `src/tape.ts` and `src/forward.ts`: add the method, written out like its neighbours, with
   `/** {@inheritDoc Arithmetic.name} */`.
4. `test/primitives.test.ts`: add its independent analytic formulas and test points to `unary` or
   `binary`; add edge-case tests for any convention. The property tests pick it up from
   `unaryNames`/`binaryNames` automatically; if it has a restricted domain, teach `safe()` in
   `test/properties.test.ts` how to keep its argument inside it.
5. Docs: the primitive lists in README.md (API overview, conventions if any), `docs/api.md`,
   `llms.txt`.
6. `npm run api:update` to refresh `etc/adjointly.api.md`, and add a changeset (`npx changeset`,
   minor).
7. `npm run check`, then `npm run mutate`, then `npm run bench` (and `bench:update` only if the
   change is meant to move the numbers).

## The gates

`npm run check` runs all of these and must be green before a commit:

| Command                 | Guards                                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `npm run format:check`  | Prettier formatting of everything (`npm run format` fixes it)                                                                                                |
| `npm run lint`          | typescript-eslint `strictTypeChecked` + `stylisticTypeChecked`, and `tsdoc/syntax` on every doc comment                                                      |
| `npm run typecheck`     | `tsc` strict with `noUncheckedIndexedAccess`, over src, tests, bench and the example; this also verifies every `// @ts-expect-error` in `test/types.test.ts` |
| `npm run test:coverage` | the Vitest suites with a 100% lines/branches/functions/statements gate on `src/`                                                                             |
| `npm run build`         | tsdown: `dist/index.js` and `dist/index.d.ts`                                                                                                                |
| `npm run publint`       | the package's `exports`/`files`/metadata are publishable                                                                                                     |
| `npm run attw`          | the types resolve for ESM consumers (`--profile esm-only`)                                                                                                   |
| `npm run api:check`     | the public API matches `etc/adjointly.api.md`; after an intended change run `npm run api:update` and commit the report                                       |

Separately, and in CI:

- `npm run mutate`: StrykerJS mutation testing of `src/`, failing below `thresholds.break` in
  `stryker.config.json`. Slow (minutes). A surviving mutant is a behaviour no test pins down: add
  the test, or, if the mutant is truly equivalent, say why in a `// Stryker disable next-line`
  comment on that line.
- `npm run bench:compare`: times the workloads and compares each mode's cost relative to plain
  evaluation with `bench/baseline.json`; warns above 1.15x the baseline ratio, fails above 1.4x.
  Ratios rather than milliseconds, because a CI runner and a laptop differ in speed by more than
  any regression worth catching. **To update the baseline** after an intended performance change:
  `npm run bench:update` on a quiet machine, check the printed numbers, commit `bench/baseline.json`.

## Doc tests

Every ` ```ts ` block in README.md, llms.txt and docs/, and every one inside a TSDoc `@example`
in `src/`, is type-checked against the source with the project's strict settings and then run
(`test/docs.test.ts`). An example that imports nothing gets the whole public API in scope. A line
ending in `//=> EXPR` asserts the statement's value (or the variable it declares) equals EXPR to a
relative 1e-12; `//=> ~EXPR` loosens that to 1e-6. EXPR may use `Math`'s members bare
(`//=> sin(1)`). Code that should not be run belongs in a ` ```text ` or ` ```sh ` block.

## Conventions

- ESM only, `.ts` import specifiers in source (tsdown rewrites them); `erasableSyntaxOnly`, so no
  enums, namespaces or parameter properties.
- TSDoc on every export, with a contract and an `@example` that the doc tests run.
- Commits: small and logical, one short line saying what changed. No trailers.
- User-visible changes get a changeset. The package stays `"private": true` and `UNLICENSED` until
  the owner decides otherwise; the release workflow is inert until then.

## Commands

```sh
npm ci                  # install
npm run check           # every gate above
npm test                # just the tests
npm run mutate          # mutation testing
npm run bench           # time the workloads
npm run example         # the production-line walkthrough
npm run format          # apply Prettier
npm run api:update      # rebuild and refresh etc/adjointly.api.md
npx changeset           # record a user-visible change
```
