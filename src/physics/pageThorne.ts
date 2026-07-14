/**
 * PHYSICS.md §7 — Novikov-Thorne thin-disk flux, Page & Thorne [R3],
 * zero-torque inner boundary at the ISCO. G = c = M = 1; flux is per unit
 * accretion rate (F_phys = F̂ Ṁ c⁶/(G M)², src/core/units.ts).
 *
 * Two independent implementations:
 *  - `fluxClosedForm`: the analytic solution (their Eq. 15n) via the roots
 *    of x³ − 3x + 2a, x = √r. Production path (feeds the LUT).
 *  - `fluxQuadrature`: direct numerical evaluation of the flux integral
 *    F = (Ṁ/4πr) (−Ω′)(E† − ΩL†)⁻² ∫ (E† − ΩL†) L†′ dr̃ with Richardson
 *    finite-difference derivatives. Validation path only (V-A6).
 */
import { richardsonDerivative } from '../core/numeric';
import { angularVelocity, iscoRadius, specificAngularMomentum, specificEnergy } from './orbits';

/** Roots x₁ ≥ x₂ ≥ x₃ of x³ − 3x + 2a = 0 (x = 2cos(⅓arccos(a) ∓ π/3), −2cos(⅓arccos(a))). */
export function radicandRoots(a: number): [number, number, number] {
  const acs = Math.acos(a) / 3;
  return [2 * Math.cos(acs - Math.PI / 3), 2 * Math.cos(acs + Math.PI / 3), -2 * Math.cos(acs)];
}

/** Analytic Page-Thorne flux F̂(r) per unit Ṁ, for r ≥ r_isco (0 at and below the ISCO edge). */
export function fluxClosedForm(a: number, r: number): number {
  const rIsco = iscoRadius(a);
  if (r <= rIsco) return 0;
  const x = Math.sqrt(r);
  const x0 = Math.sqrt(rIsco);
  const roots = radicandRoots(a);

  let bracket = x - x0 - 1.5 * a * Math.log(x / x0);
  for (let i = 0; i < 3; i++) {
    const xi = roots[i]!;
    const xj = roots[(i + 1) % 3]!;
    const xk = roots[(i + 2) % 3]!;
    // At a = 0 one root is exactly 0 with coefficient limit 0 (numerator (xᵢ−a)² → 0 faster).
    if (Math.abs(xi) < 1e-12) continue;
    const ci = (3 * (xi - a) * (xi - a)) / (xi * (xi - xj) * (xi - xk));
    bracket -= ci * Math.log((x - xi) / (x0 - xi));
  }

  const w = x * x * x - 3 * x + 2 * a; // = r^{3/2} − 3r^{1/2} + 2a
  const f = (3 / (2 * x * x * w)) * bracket;
  return f / (4 * Math.PI * r);
}

/** Adaptive Simpson quadrature. */
function adaptiveSimpson(
  fn: (x: number) => number,
  lo: number,
  hi: number,
  tol: number,
  depth = 24,
): number {
  const simpson = (x0: number, x2: number): number => {
    const x1 = 0.5 * (x0 + x2);
    return ((x2 - x0) / 6) * (fn(x0) + 4 * fn(x1) + fn(x2));
  };
  const recurse = (x0: number, x2: number, whole: number, eps: number, d: number): number => {
    const x1 = 0.5 * (x0 + x2);
    const left = simpson(x0, x1);
    const right = simpson(x1, x2);
    if (d <= 0 || Math.abs(left + right - whole) <= 15 * eps) {
      return left + right + (left + right - whole) / 15;
    }
    return recurse(x0, x1, left, eps / 2, d - 1) + recurse(x1, x2, right, eps / 2, d - 1);
  };
  const whole = simpson(lo, hi);
  return recurse(lo, hi, whole, tol, depth);
}

/** Numerical-quadrature Page-Thorne flux (validation only; slow). */
export function fluxQuadrature(a: number, r: number): number {
  const rIsco = iscoRadius(a);
  if (r <= rIsco) return 0;

  const eMinusOmegaL = (rr: number): number =>
    specificEnergy(a, rr) - angularVelocity(a, rr) * specificAngularMomentum(a, rr);
  const integrand = (rr: number): number =>
    eMinusOmegaL(rr) * richardsonDerivative((x) => specificAngularMomentum(a, x), rr, 1e-4 * rr);

  const integral = adaptiveSimpson(integrand, rIsco, r, 1e-13 * Math.max(1, r - rIsco));
  // Ω′(r) = −(3/2) r^{1/2} / (r^{3/2} + a)² (analytic).
  const dOmega = (-1.5 * Math.sqrt(r)) / Math.pow(Math.pow(r, 1.5) + a, 2);
  const denom = eMinusOmegaL(r);
  return ((-dOmega / (denom * denom)) * integral) / (4 * Math.PI * r);
}
