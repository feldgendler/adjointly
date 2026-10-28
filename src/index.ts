/**
 * adjointly: dependency-free automatic differentiation for scalar float64 programs.
 *
 * Write a function once against {@link Arithmetic}, then run it on plain numbers ({@link plain},
 * {@link evaluate}), in reverse mode for gradients ({@link Tape}, {@link gradient}), or in forward
 * mode for directional derivatives ({@link Forward}, {@link jvp}).
 *
 * @example
 * ```ts
 * import { type Arithmetic, gradient } from 'adjointly';
 *
 * function f<T>(A: Arithmetic<T>, [x, y]: readonly T[]): T {
 *   return A.mul(A.sin(x!), A.exp(y!));
 * }
 * gradient(f, [1, 0]); //=> { value: sin(1), gradient: [cos(1), sin(1)] }
 * ```
 *
 * @packageDocumentation
 */

export { type Arithmetic, type ScalarFunction, plain } from './arithmetic.ts';
export {
  type BinaryName,
  type BinaryPrimitive,
  type UnaryName,
  type UnaryPrimitive,
  binaryNames,
  primitives,
  unaryNames,
} from './primitives.ts';
export { type Mark, type TapeOptions, type Var, Tape } from './tape.ts';
export { type Dual, type ForwardOptions, Forward } from './forward.ts';
export { dot, logsumexp, softmax, sum } from './composites.ts';
export { type GradientResult, type JvpResult, evaluate, gradient, jvp } from './convenience.ts';
export {
  type CheckGradientOptions,
  type GradientCheck,
  DEFAULT_STEP,
  checkGradient,
  finiteDifference,
} from './check.ts';
