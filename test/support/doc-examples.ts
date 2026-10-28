/*
 * Extracts the code examples from the documentation so they can be type-checked and run as tests.
 *
 * Sources: every ```ts fence in README.md, llms.txt and docs/**\/*.md, and every ```ts fence inside a
 * TSDoc `@example` in src/. An example that imports nothing from 'adjointly' gets the whole public
 * API in scope (TSDoc examples read better without an import line).
 *
 * Assertions: a line ending in `//=> EXPR` asserts that the statement's value (or the variable it
 * declares) deep-equals EXPR to 1e-12 relative; `//=> ~EXPR` loosens that to 1e-6. EXPR is evaluated
 * with Math's members in scope, so `//=> sin(3)` and `//=> [cos(1), sin(1)]` work. An assertion must
 * fit on one line.
 */

import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

export interface DocExample {
  /** `path:line` of the fence's first code line, for test names and diagnostics. */
  readonly where: string;
  /** The TypeScript source of the example. */
  readonly code: string;
  /** 1-based line in the source file of the example's first line. */
  readonly line: number;
  readonly file: string;
}

export const root = join(import.meta.dirname, '..', '..');

function markdownFiles(): string[] {
  const docs = join(root, 'docs');
  const pages = readdirSync(docs, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('.md'))
    .map((f) => join(docs, f));
  return [join(root, 'README.md'), join(root, 'llms.txt'), ...pages.sort()];
}

function sourceFiles(): string[] {
  const src = join(root, 'src');
  return readdirSync(src)
    .filter((f) => f.endsWith('.ts'))
    .sort()
    .map((f) => join(src, f));
}

const FENCE_OPEN = /^\s*```(ts|typescript)\s*$/;
const FENCE_CLOSE = /^\s*```\s*$/;

function fences(file: string, lines: readonly string[]): DocExample[] {
  const out: DocExample[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (!FENCE_OPEN.test(lines[i]!)) continue;
    const start = i + 1;
    let end = start;
    while (end < lines.length && !FENCE_CLOSE.test(lines[end]!)) end++;
    const rel = relative(root, file);
    out.push({
      file: rel,
      line: start + 1,
      where: `${rel}:${String(start + 1)}`,
      code: lines.slice(start, end).join('\n'),
    });
    i = end;
  }
  return out;
}

/** Every example in the docs and the TSDoc, in a stable order. */
export function docExamples(): DocExample[] {
  const out: DocExample[] = [];
  for (const file of markdownFiles()) {
    out.push(...fences(file, readFileSync(file, 'utf8').split('\n')));
  }
  for (const file of sourceFiles()) {
    // Strip the ` * ` gutter of doc comments, keeping line numbers.
    const raw = readFileSync(file, 'utf8').split('\n');
    let inExample = false;
    const kept = raw.map((r) => {
      const l = r.replace(/^\s*\* ?/, '');
      if (/\*\/\s*$/.test(r)) inExample = false;
      else if (/^@example\b/.test(l)) inExample = true;
      else if (/^@\w/.test(l)) inExample = false;
      return inExample ? l : '';
    });
    out.push(...fences(file, kept));
  }
  return out;
}

const ASSERT =
  /^(\s*)(?:(const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*)?(.*?);?\s*\/\/=>\s*(~?)(.*?)\s*$/;

const MATH_NAMES = new Set(Object.getOwnPropertyNames(Math));

/**
 * Rewrite each `X; //=> EXPR` line of an example's TypeScript into `__check(X, () => EXPR, ...)`,
 * with the Math members EXPR mentions in scope. `line` is reported back to `__check` as the
 * example's own 1-based line number.
 */
export function instrumentAssertions(code: string): string {
  return code
    .split('\n')
    .map((line, i) => {
      const m = ASSERT.exec(line);
      if (!m) return line;
      const [, indent, decl, name, expr, loose, expected] = m;
      const used = [...new Set(expected!.match(/\b[A-Za-z_$][\w$]*\b/g) ?? [])].filter((id) =>
        MATH_NAMES.has(id),
      );
      const scope = used.length ? `const { ${used.join(', ')} } = Math; ` : '';
      const want = `() => { ${scope}return (${expected!}); }`;
      const args = `${want}, ${loose ? 'true' : 'false'}, ${String(i + 1)}`;
      return decl
        ? `${indent!}${decl} ${name!} = ${expr!}; __check(${name!}, ${args});`
        : `${indent!}__check((${expr!}), ${args});`;
    })
    .join('\n');
}

/** Rewrite the imports from 'adjointly' in transpiled JavaScript into destructuring of `__lib`. */
export function rewriteImports(js: string): string {
  return js
    .split('\n')
    .map((line) => {
      const imp = /^\s*import\s*\{([^}]*)\}\s*from\s*['"]adjointly['"];?\s*$/.exec(line);
      if (imp) {
        return `const {${imp[1]!.replace(/\bas\b/g, ':')}} = __lib; __names(${JSON.stringify(imp[1])});`;
      }
      if (/^\s*import\b/.test(line)) throw new Error(`unsupported import: ${line}`);
      return line;
    })
    .join('\n');
}

function close(a: number, e: number, rtol: number): boolean {
  if (Object.is(a, e)) return true;
  if (Number.isNaN(a) || Number.isNaN(e)) return false;
  return Math.abs(a - e) <= rtol * Math.max(1, Math.abs(e));
}

/** Deep approximate equality: numbers within `rtol`, arrays and typed arrays elementwise. */
export function deepClose(actual: unknown, expected: unknown, rtol: number): boolean {
  if (typeof expected === 'number')
    return typeof actual === 'number' && close(actual, expected, rtol);
  if (Array.isArray(expected)) {
    const a = ArrayBuffer.isView(actual) ? Array.from(actual as Float64Array) : actual;
    return (
      Array.isArray(a) &&
      a.length === expected.length &&
      expected.every((e, i) => deepClose(a[i], e, rtol))
    );
  }
  if (expected !== null && typeof expected === 'object') {
    if (actual === null || typeof actual !== 'object') return false;
    const ek = Object.keys(expected).sort();
    const ak = Object.keys(actual).sort();
    return (
      ek.join() === ak.join() &&
      ek.every((k) =>
        deepClose(
          (actual as Record<string, unknown>)[k],
          (expected as Record<string, unknown>)[k],
          rtol,
        ),
      )
    );
  }
  return Object.is(actual, expected);
}
