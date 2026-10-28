# Working on adjointly

Instructions for agents (and people) changing this library. Users of the library want
[README.md](README.md) and [docs/](docs/README.md) instead.

## Layout

| Path                        | What it is                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------- |
| `src/primitives.ts`         | **The** table of primitives: every value formula and every local partial derivative                |
| `src/arithmetic.ts`         | `Arithmetic<T>`, `ScalarFunction`, and `plain` (the table's value functions)                       |
| `src/tape.ts`               | `Tape`, reverse mode: typed-array storage, recording, the backward sweep, mark/rewind              |
| `src/forward.ts`            | `Forward`, forward mode: a value and a tangent per slot                                            |
| `src/composites.ts`         | `sum`, `dot`, `logsumexp`, `softmax`, generic over any `Arithmetic`                                |
| `src/convenience.ts`        | `evaluate`, `gradient`, `jvp`                                                                      |
| `src/check.ts`              | `finiteDifference`, `checkGradient`                                                                |
| `src/capacity.ts`           | the `capacity` option check shared by `Tape` and `Forward`                                         |
| `src/index.ts`              | the public API: everything exported from the package root                                          |
| `test/`                     | Vitest suites, helpers in `test/support/`; `invariants.test.ts` reads the source itself            |
| `bench/`                    | tinybench workloads (`workloads.ts`), the measurement (`measure.ts`), the A/B comparison (`ab.ts`) |
| `examples/production-line/` | the worked example; run by `npm run example`, tested by `test/example.test.ts`                     |
| `docs/`, `llms.txt`         | user documentation, shipped in the package                                                         |
| `etc/adjointly.api.md`      | API Extractor's report of the public API, committed                                                |
| `.changeset/`               | pending changesets                                                                                 |

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
7. `npm run check`, then `npm run mutate`, then `npm run bench:ab` against the commit before
   (see below).

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
- `npm run bench:ab -- --base <dir>`: benchmarks this checkout against another one in
  alternating rounds on the same machine (`bench/ab.ts`), so that the machine's speed and load
  cancel out; there is no stored baseline. Each round runs `bench/measure.ts` once per side, as
  separate processes, against that side's `src/`; each case's time is the median over the rounds.
  Head / base above `BENCH_WARN` (default 1.05) is a warning annotation, above `BENCH_FAIL`
  (default 1.15) an error and exit code 1. CI's `bench` job runs it with 5 rounds: on a pull
  request against the PR's base commit, on a push to main against the commit main was at before
  the push (`github.event.before`); a push that creates a branch has no base and skips. Locally:
  `git worktree add .bench-base <ref>` (ignored by git and the linters), then
  `npm run bench:ab -- --base .bench-base`, then `git worktree remove .bench-base`. On a GitHub
  runner two identical sides came out within 1.2% of each other; a busy laptop shows up to ~10%,
  so judge a local run with that in mind.
- `npm run bench`: times this checkout's workloads and prints each mode's cost per node and as a
  multiple of plain evaluation. The workloads (`bench/workloads.ts`) take the library as a
  parameter so that `ab.ts` can run the same code against both sides; they use only
  `Arithmetic`, `logsumexp`, `plain`, `Tape` and `Forward`, which every version has.

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
- User-visible changes get a changeset. The license is MIT (`LICENSE`). The package stays
  `"private": true` until the owner decides otherwise; the release workflow is inert until then.
- Versions: `package.json` stays at 0.0.0 until the first release, and the pending first-release
  changeset is a minor bump, so the first release is 0.1.0. Until 1.0, a breaking change is a
  minor bump and anything else a patch. `.changeset/config.json` sets
  `privatePackages.version: true`; without it Changesets ignores a private package and versions
  nothing.

## Releasing

`.github/workflows/release.yml` runs only when the repository variable `RELEASE_ENABLED` is
`true`, and refuses to publish while `package.json` is private. Before enabling it:

- Remove `"private": true` (the license, MIT, is already in place).
- Configure npm trusted publishing (OIDC) for this repository and workflow on npmjs.com; no npm
  token is stored.
- **Provenance and private repositories:** npm may refuse provenance attestations for packages
  published from private GitHub repositories. Either make the repository public first, or check
  npm's current rules and, if needed, drop `"provenance": true` from `publishConfig`.

With pending changesets, the workflow opens a "Version packages" pull request
(`npm run version-packages`: version bump and CHANGELOG); merging it publishes. That pull request
is opened with the workflow's `GITHUB_TOKEN`, and GitHub does not start other workflows for
events caused by that token, so **CI does not run on it**. Before merging, run CI on it by hand
(push an empty commit to its branch, or close and reopen it), or pass `changesets/action` the
token of a GitHub App or a fine-grained personal access token in its `github-token` input (v2
reads no token from the environment), whose pull requests do trigger CI.

## Commands

````sh
npm ci                  # install
npm run check           # every gate above
npm test                # just the tests
npm run mutate          # mutation testing
npm run bench           # time the workloads
npm run bench:ab -- --base <dir>   # compare with another checkout
npm run example         # the production-line walkthrough
npm run format          # apply Prettier
npm run api:update      # rebuild and refresh etc/adjointly.api.md
npx changeset           # record a user-visible change
```
````
