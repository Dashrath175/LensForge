/**
 * V-A5 (VALIDATION.md section 1): ISCO and photon-orbit radii -- independent
 * numerical root-finding of the defining conditions vs the BPT closed forms.
 */
import { describe, expect, it } from 'vitest';
import { richardsonDerivative } from '../../../src/core/numeric';
import {
  angularVelocity,
  circularUt,
  iscoRadius,
  photonOrbitRadius,
  specificAngularMomentum,
  specificEnergy,
} from '../../../src/physics/orbits';
import { radicandRoots } from '../../../src/physics/pageThorne';

const SPINS = [0, 0.25, 0.5, 0.75, 0.9, 0.998];

function bisectRoot(f: (s: number) => number, lo: number, hi: number): number {
  let fLo = f(lo);
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    const fMid = f(mid);
    if (Math.sign(fMid) === Math.sign(fLo)) {
      lo = mid;
      fLo = fMid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

describe('V-A5: characteristic radii vs closed forms', () => {
  it('ISCO: first root of the marginal-stability quartic s^4 - 6s^2 + 8as - 3a^2 (s = sqrt(r)) matches BPT', () => {
    for (const a of SPINS) {
      const f = (s: number): number => s ** 4 - 6 * s * s + 8 * a * s - 3 * a * a;
      // Scan upward from s = 1 for the first sign change (prograde branch).
      let sLo = 1.0;
      let sHi = 3.0;
      for (let s = 1.0; s < 3.0; s += 0.01) {
        if (Math.sign(f(s)) !== Math.sign(f(sLo))) {
          sHi = s;
          break;
        }
        sLo = s;
      }
      const rNumeric = bisectRoot(f, sLo, sHi) ** 2;
      expect(Math.abs(rNumeric - iscoRadius(a)) / iscoRadius(a)).toBeLessThan(1e-10);
    }
  });

  it('photon orbit: root of s^3 - 3s + 2a matches the trigonometric closed form and x1^2', () => {
    for (const a of SPINS) {
      const rNumeric = bisectRoot((s) => s ** 3 - 3 * s + 2 * a, 1.0, 2.0) ** 2;
      const rClosed = photonOrbitRadius(a, true);
      const [x1] = radicandRoots(a);
      expect(Math.abs(rNumeric - rClosed) / rClosed).toBeLessThan(1e-10);
      expect(Math.abs(x1 * x1 - rClosed) / rClosed).toBeLessThan(1e-12);
    }
  });

  it('Schwarzschild exact values: r_isco = 6, E_isco = 2*sqrt(2)/3, L_isco = 2*sqrt(3), r_ph = 3', () => {
    expect(iscoRadius(0)).toBeCloseTo(6, 12);
    expect(specificEnergy(0, 6)).toBeCloseTo((2 * Math.sqrt(2)) / 3, 13);
    expect(specificAngularMomentum(0, 6)).toBeCloseTo(2 * Math.sqrt(3), 13);
    expect(photonOrbitRadius(0, true)).toBeCloseTo(3, 13);
    expect(photonOrbitRadius(0, false)).toBeCloseTo(3, 13);
  });

  it('extreme prograde limits: r_isco -> 1 and r_ph -> 1 as a -> 1', () => {
    expect(iscoRadius(0.9999999)).toBeLessThan(1.02);
    expect(photonOrbitRadius(0.9999999, true)).toBeLessThan(1.01);
  });

  it('E declines to a minimum at the ISCO (marginal stability semantics)', () => {
    for (const a of SPINS) {
      const rIsco = iscoRadius(a);
      const dEdr = (r: number): number =>
        richardsonDerivative((x) => specificEnergy(a, x), r, 1e-4 * r);
      expect(Math.abs(dEdr(rIsco))).toBeLessThan(1e-6);
      expect(dEdr(rIsco * 1.5)).toBeGreaterThan(0);
    }
  });

  it('Schwarzschild kinematics: Omega = r^(-3/2) and u^t = 1/sqrt(1 - 3/r)', () => {
    for (const r of [6, 10, 50]) {
      expect(angularVelocity(0, r)).toBeCloseTo(Math.pow(r, -1.5), 14);
      expect(circularUt(0, r)).toBeCloseTo(1 / Math.sqrt(1 - 3 / r), 13);
    }
  });
});
