import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    // Doc examples import from the package name; resolve it to the source, not dist/.
    alias: { adjointly: fileURLToPath(new URL('./src/index.ts', import.meta.url)) },
  },
  test: {
    include: ['test/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      reporter: ['text', 'html'],
      thresholds: { lines: 100, branches: 100, functions: 100, statements: 100 },
    },
  },
});
