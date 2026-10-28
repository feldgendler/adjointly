/*
 * Invariants of the source and the package that no behavioural test would notice breaking: they
 * read the source text, the docs and package.json.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import ts from 'typescript';
import { describe, expect, test } from 'vitest';
import { constantOperandOps } from '../src/arithmetic.ts';
import { binaryNames, primitives, unaryNames } from '../src/index.ts';

const root = join(import.meta.dirname, '..');
const read = (path: string): string => readFileSync(join(root, path), 'utf8');
const parse = (path: string): ts.SourceFile =>
  ts.createSourceFile(path, read(path), ts.ScriptTarget.ES2022, true);

/** The doc comment directly before a node, as written. */
function docOf(node: ts.Node, file: ts.SourceFile): string | undefined {
  const ranges = ts.getLeadingCommentRanges(file.text, node.getFullStart()) ?? [];
  const last = ranges.at(-1);
  const text = last && file.text.slice(last.pos, last.end);
  return text?.startsWith('/**') ? text : undefined;
}

/** Each member's doc comment, by name, from an interface or a class. */
function memberDocs(path: string, container: string): Map<string, string | undefined> {
  const file = parse(path);
  const out = new Map<string, string | undefined>();
  file.forEachChild((node) => {
    if (
      (ts.isInterfaceDeclaration(node) || ts.isClassDeclaration(node)) &&
      node.name?.text === container
    ) {
      for (const m of node.members) {
        if ((ts.isMethodSignature(m) || ts.isMethodDeclaration(m)) && ts.isIdentifier(m.name)) {
          out.set(m.name.text, docOf(m, file));
        }
      }
    }
  });
  return out;
}

/** Each method's body text, by name, from a class. */
function methodBodies(path: string, container: string): Map<string, string> {
  const file = parse(path);
  const out = new Map<string, string>();
  file.forEachChild((node) => {
    if (ts.isClassDeclaration(node) && node.name?.text === container) {
      for (const m of node.members) {
        if (ts.isMethodDeclaration(m) && ts.isIdentifier(m.name) && m.body) {
          out.set(m.name.text, m.body.getText(file));
        }
      }
    }
  });
  return out;
}

const names = [...unaryNames, ...binaryNames];

describe('Tape and Forward', () => {
  const contract = memberDocs('src/arithmetic.ts', 'Arithmetic');

  test.each(['Tape', 'Forward'])(
    "%s documents every Arithmetic method with the interface's own text",
    (cls) => {
      // TypeScript shows `{@inheritDoc}` literally in editor hovers, so the text is copied, and
      // kept identical here.
      const docs = memberDocs(`src/${cls === 'Tape' ? 'tape' : 'forward'}.ts`, cls);
      expect(contract.size).toBe(names.length + 2 + Object.keys(constantOperandOps).length);
      for (const [name, doc] of contract) {
        expect(doc, `Arithmetic.${name}`).toBeDefined();
        expect(docs.get(name), `${cls}.${name}`).toBe(doc);
      }
    },
  );

  test.each(['Tape', 'Forward'])(
    '%s computes each primitive from its own table entry and nothing else',
    (cls) => {
      const bodies = methodBodies(`src/${cls === 'Tape' ? 'tape' : 'forward'}.ts`, cls);
      for (const name of names) {
        const body = bodies.get(name);
        expect(body, `${cls}.${name}`).toBeDefined();
        const entries = new Set([...body!.matchAll(/\bP\.(\w+)\./g)].map((m) => m[1]));
        expect([...entries], `${cls}.${name} reads the table`).toEqual([name]);
        expect(body, `${cls}.${name} must not compute a formula itself`).not.toMatch(/\bMath\./);
      }
      for (const [name, { of }] of Object.entries(constantOperandOps)) {
        const body = bodies.get(name);
        expect(body, `${cls}.${name}`).toBeDefined();
        const entries = new Set([...body!.matchAll(/\bP\.(\w+)\./g)].map((m) => m[1]));
        expect([...entries], `${cls}.${name} reads the table`).toEqual([of]);
        expect(body, `${cls}.${name} must not compute a formula itself`).not.toMatch(/\bMath\./);
      }
    },
  );
});

describe('the package', () => {
  test('has no runtime dependencies', () => {
    const pkg = JSON.parse(read('package.json')) as Record<string, unknown>;
    expect(pkg.dependencies ?? {}).toEqual({});
    expect(pkg.peerDependencies ?? {}).toEqual({});
    expect(pkg.optionalDependencies ?? {}).toEqual({});
  });
});

describe('the docs list exactly the primitives in the table', () => {
  const table = Object.keys(primitives).sort();
  const words = (list: string): string[] =>
    list
      .split(/[,\s`]+/)
      .filter(Boolean)
      .sort();

  test('README.md', () => {
    const m = /one method per\s+primitive: `([^`]+)`/.exec(read('README.md'));
    expect(m, 'the primitive list in the API overview').not.toBeNull();
    expect(words(m![1]!)).toEqual(table);
  });

  test('llms.txt', () => {
    const m = /Primitives \(all on `Arithmetic`\): ([^.]+)\./.exec(read('llms.txt'));
    expect(m).not.toBeNull();
    expect(words(m![1]!)).toEqual(table);
  });

  test('docs/api.md', () => {
    const text = read('docs/api.md');
    const unary = /\| `((?:\w+, )+\w+)` `\(a: T\): T`/.exec(text);
    const binary = /\| `((?:\w+, )+\w+)` `\(a: T, b: T\): T`/.exec(text);
    expect(unary).not.toBeNull();
    expect(binary).not.toBeNull();
    expect(words(unary![1]!)).toEqual([...unaryNames].sort());
    expect(words(binary![1]!)).toEqual([...binaryNames].sort());
  });
});

test.each(['README.md', 'docs/api.md', 'llms.txt'])(
  '%s names every operation with a plain-number operand',
  (path) => {
    const text = read(path);
    for (const name of Object.keys(constantOperandOps)) expect(text, name).toContain(`${name}(`);
  },
);
