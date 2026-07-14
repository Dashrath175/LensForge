/**
 * V-A6 (VALIDATION.md section 1): Page-Thorne disk flux -- analytic closed
 * form vs independent numerical quadrature of the flux integral, plus the
 * Newtonian large-radius limit and boundary behavior.
 */
import { describe, expect, it } from 'vitest';
import { iscoRadius } from '../../../src/physics/orbits';
import { fluxClosedForm, fluxQuadrature } from '../../../src/physics/pageThorne';

const SPINS = [0, 0.5, 0.9, 0.998];

describe('V-A6: Page-Thorne flux', () => {
  it('closed form matches independent quadrature to rel 1e-6', () => {
    for (const a of SPINS) {
      const rIsco = iscoRadius(a);
      for (const ratio of [1.05, 1.2, 2, 5, 20, 100]) {
        const r = rIsco * ratio;
        const closed = fluxClosedForm(a, r);
        const quad = fluxQuadrature(a, r);
        expect(closed).toBeGreaterThan(0);
        expect(Math.abs(closed - quad) / closed).toBeLessThan(1e-6);
      }
    }
  });

  it('has the correct Newtonian asymptotics at large radius (a = 0)', () => {
    // Leading order F -> 3/(8 pi r^3), but the boundary constant is NOT the
    // naive sqrt(6): the zero-torque log terms shift it to
    //   C = sqrt(6) + (sqrt(3)/2) ln((sqrt(6)+sqrt(3))/(sqrt(6)-sqrt(3))) ~= 3.976,
    // so the naive profile is recovered only as O(r^-1/2).
    const C = Math.sqrt(6) + (Math.sqrt(3) / 2) * Math.log((Math.sqrt(6) + Math.sqrt(3)) / (Math.sqrt(6) - Math.sqrt(3)));
    const corrected = (r: number): number => (3 / (8 * Math.PI * r ** 3)) * (1 - C / Math.sqrt(r));
    const naive = (r: number): number => (3 / (8 * Math.PI * r ** 3)) * (1 - Math.sqrt(6 / r));

    const relErrCorrected = (r: number): number =>
      Math.abs(fluxClosedForm(0, r) - corrected(r)) / corrected(r);
    const relErrNaive = (r: number): number =>
      Math.abs(fluxClosedForm(0, r) - naive(r)) / naive(r);

    // Against the log-corrected boundary constant: next correction is O(1/r).
    expect(relErrCorrected(1e5)).toBeLessThan(1e-4);
    expect(relErrCorrected(1e6)).toBeLessThan(1e-5);
    // Against the naive profile: residual is (C - sqrt(6))/sqrt(r), converging as r^-1/2.
    expect(relErrNaive(1e6)).toBeLessThan(2 / Math.sqrt(1e6));
    expect(relErrNaive(1e6) / relErrNaive(1e4)).toBeLessThan(0.15);
  });

  it('vanishes at and below the ISCO (zero-torque boundary) and is positive above it', () => {
    for (const a of SPINS) {
      const rIsco = iscoRadius(a);
      expect(fluxClosedForm(a, rIsco)).toBe(0);
      expect(fluxClosedForm(a, rIsco * 0.9)).toBe(0);
      expect(fluxClosedForm(a, rIsco * (1 + 1e-6))).toBeGreaterThan(0);
    }
  });
});
