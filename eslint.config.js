// @ts-check
import js from '@eslint/js';
import tsdoc from 'eslint-plugin-tsdoc';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist/', 'coverage/', 'reports/', 'temp/', '.stryker-tmp/'] },
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    plugins: { tsdoc },
    rules: {
      'tsdoc/syntax': 'error',
      // Indexing a typed array yields `number | undefined` under noUncheckedIndexedAccess even
      // for indices the code has just bounds-checked; `!` is the idiom for those reads and
      // costs nothing at run time, where a guard per access would sit in the hot loops.
      '@typescript-eslint/no-non-null-assertion': 'off',
      // Numbers in error messages are the common case here.
      '@typescript-eslint/restrict-template-expressions': ['error', { allowNumber: true }],
    },
  },
  {
    files: ['**/*.js', '**/*.mjs'],
    ...tseslint.configs.disableTypeChecked,
  },
);
