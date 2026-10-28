/*
 * Doc tests: every ```ts example in README.md, llms.txt, docs/ and the TSDoc `@example` blocks is
 * type-checked against the source under the project's strict settings, then run, with its
 * `//=> EXPR` lines asserted. See test/support/doc-examples.ts for the conventions.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import ts from 'typescript';
import { describe, expect, test } from 'vitest';
import * as lib from '../src/index.ts';
import {
  deepClose,
  docExamples,
  instrumentAssertions,
  rewriteImports,
  root,
} from './support/doc-examples.ts';

const examples = docExamples();

const valueExports = Object.keys(lib).sort();
// The type-only exports, from the committed API report (which `npm run api:check` keeps current).
const typeExports = [
  ...readFileSync(join(root, 'etc', 'adjointly.api.md'), 'utf8').matchAll(
    /^export (?:type|interface) (\w+)/gm,
  ),
]
  .map((m) => m[1]!)
  .sort();

const importsLib = (code: string): boolean => /from\s+['"]adjointly['"]/.test(code);

/** The example as a module, with the whole API imported when it imports nothing itself. */
function asModule(code: string): { text: string; offset: number } {
  if (importsLib(code)) return { text: `${code}\nexport {};\n`, offset: 0 };
  const prelude =
    `import { ${valueExports.join(', ')}, ` +
    `${typeExports.map((t) => `type ${t}`).join(', ')} } from 'adjointly';`;
  return { text: `${prelude}\n${code}\nexport {};\n`, offset: 1 };
}

test('the type exports come from the API report', () => {
  expect(typeExports).toContain('Arithmetic');
  expect(typeExports).toContain('Var');
  expect(typeExports.some((t) => valueExports.includes(t))).toBe(false);
});

test('there are doc examples in every kind of source', () => {
  const files = new Set(examples.map((e) => e.file.split('/')[0]));
  expect(files).toContain('README.md');
  expect(files).toContain('llms.txt');
  expect(files).toContain('docs');
  expect(files).toContain('src');
});

test.skipIf(process.env.ADJOINTLY_MUTATION)(
  'every doc example type-checks',
  { timeout: 120_000 },
  () => {
    const config = ts.getParsedCommandLineOfConfigFile(
      join(root, 'tsconfig.json'),
      {},
      {
        ...ts.sys,
        onUnRecoverableConfigFileDiagnostic: (d) => {
          throw new Error(ts.flattenDiagnosticMessageText(d.messageText, '\n'));
        },
      },
    );
    if (!config) throw new Error('cannot read tsconfig.json');
    const options: ts.CompilerOptions = {
      ...config.options,
      noEmit: true,
      noUnusedLocals: false,
      noUnusedParameters: false,
      paths: { adjointly: ['./src/index.ts'] },
    };
    const virtual = new Map<
      string,
      { text: string; offset: number; where: string; line: number }
    >();
    examples.forEach((e, i) => {
      virtual.set(join(root, 'test', `__doc_example_${String(i)}.ts`), {
        ...asModule(e.code),
        where: e.where,
        line: e.line,
      });
    });
    const host = ts.createCompilerHost(options);
    const getSourceFile = host.getSourceFile.bind(host);
    const fileExists = host.fileExists.bind(host);
    const readFile = host.readFile.bind(host);
    host.fileExists = (f) => virtual.has(f) || fileExists(f);
    host.readFile = (f) => virtual.get(f)?.text ?? readFile(f);
    host.getSourceFile = (f, lang, ...rest) => {
      const v = virtual.get(f);
      return v ? ts.createSourceFile(f, v.text, lang) : getSourceFile(f, lang, ...rest);
    };
    const program = ts.createProgram([...virtual.keys()], options, host);
    const problems: string[] = [];
    for (const d of ts.getPreEmitDiagnostics(program)) {
      const message = ts.flattenDiagnosticMessageText(d.messageText, '\n');
      const v = d.file && virtual.get(d.file.fileName);
      if (v && d.start !== undefined && d.file) {
        const { line } = d.file.getLineAndCharacterOfPosition(d.start);
        problems.push(`${v.where.split(':')[0]!}:${String(v.line + line - v.offset)}: ${message}`);
      } else {
        problems.push(`${d.file?.fileName ?? '(global)'}: ${message}`);
      }
    }
    expect(problems).toEqual([]);
  },
);

describe('every doc example runs and its //=> assertions hold', () => {
  for (const e of examples) {
    test(e.where, () => {
      const { text } = asModule(instrumentAssertions(e.code));
      const js = ts.transpileModule(text, {
        compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
      }).outputText;
      const body = rewriteImports(js.replace(/^export \{\};\s*$/m, ''));
      const failures: string[] = [];
      const check = (actual: unknown, want: () => unknown, loose: boolean, line: number): void => {
        const expected = want();
        if (!deepClose(actual, expected, loose ? 1e-6 : 1e-12)) {
          failures.push(
            `${e.file}:${String(e.line + line - 1)}: got ${JSON.stringify(actual, numbers)}, ` +
              `expected ${JSON.stringify(expected, numbers)}`,
          );
        }
      };
      const names = (list: string): void => {
        for (const raw of list.split(',')) {
          const name = raw
            .replace(/\btype\b/, '')
            .split(/\bas\b/)[0]!
            .trim();
          if (name && !(name in lib) && !typeExports.includes(name)) {
            failures.push(`${e.where}: 'adjointly' has no export ${name}`);
          }
        }
      };
      // Running documentation code is the point of this test, so it is evaluated, not imported.
      // eslint-disable-next-line @typescript-eslint/no-implied-eval
      const run = new Function('__lib', '__check', '__names', body) as (...args: unknown[]) => void;
      run(lib, check, names);
      expect(failures).toEqual([]);
    });
  }
});

function numbers(_key: string, v: unknown): unknown {
  if (typeof v === 'number' && !Number.isFinite(v)) return String(v);
  if (ArrayBuffer.isView(v)) return Array.from(v as Float64Array);
  return v;
}

test('the doc-test machinery itself catches a wrong assertion', () => {
  const code = instrumentAssertions('const x = 1 + 1; //=> 3\nx * 2; //=> ~4.000001');
  const seen: boolean[] = [];
  const check = (actual: unknown, want: () => unknown, loose: boolean): void => {
    seen.push(deepClose(actual, want(), loose ? 1e-6 : 1e-12));
  };
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  (new Function('__check', code) as (c: typeof check) => void)(check);
  expect(seen).toEqual([false, true]);
  expect(deepClose({ a: [1, 2] }, { a: [1, 2], b: 1 }, 0)).toBe(false);
  expect(deepClose(Float64Array.from([1, 2]), [1, 2], 0)).toBe(true);
  expect(deepClose(NaN, NaN, 0)).toBe(true);
  expect(deepClose(1, NaN, 0)).toBe(false);
  expect(() => rewriteImports("import { x } from 'other';")).toThrow(/unsupported import/);
});
