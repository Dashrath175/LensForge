/** Shared numerical helpers (float64 oracle side). */

/**
 * Richardson-extrapolated central difference, O(h⁴). Used for validation-side
 * derivatives where analytic forms would add derivation risk (DEVELOPMENT.md §5).
 */
export function richardsonDerivative(fn: (x: number) => number, x: number, h: number): number {
  const d1 = (fn(x + h) - fn(x - h)) / (2 * h);
  const d2 = (fn(x + h / 2) - fn(x - h / 2)) / h;
  return (4 * d2 - d1) / 3;
}
