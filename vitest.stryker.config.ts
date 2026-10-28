import { defineConfig, mergeConfig } from 'vitest/config';
import base from './vitest.config.ts';

// Mutation testing runs the suite against Stryker's instrumented copy of src/, which does not
// type-check; the flag lets the doc tests skip their type-check (Stryker's TypeScript checker
// covers that) while still running every example.
export default mergeConfig(base, defineConfig({ test: { env: { ADJOINTLY_MUTATION: '1' } } }));
