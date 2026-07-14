/**
 * V-A2 (conservation), V-A3 (critical impact parameter), V-A4 (weak-field
 * deflection) -- VALIDATION.md section 1 -- plus disk-crossing refinement consistency.
 */
import { describe, expect, it } from 'vitest';
import { IPH, IR } from '../../../src/physics/hamiltonian';
import { integrateGeodesic } from '../../../src/physics/integrator';
import { createMetric, evalMetric } from '../../../src/physics/metric';
import { iscoRadius } from '../../../src/physics/orbits';
import { equatorialPhoton, photonFromTetrad } from '../../../src/physics/photon';
import { orthonormalTetrad, zamoFourVelocity } from '../../../src/physics/tetrad';

const B_CRIT_SCHWARZSCHILD = 3 * Math.sqrt(3);

// Accumulated truncation floor at rtol 1e-13 is ~5e-16/step over O(2000)
// steps; measured worst 1.0e-12. The bound below is that floor with margin.
const DRIFT_BOUND = 5e-12;

describe('V-A2: conservation drift over the standard ray set', () => {
  it('max normalized |H| and relative Q drift < 5e-12 at rtol 1e-13', () => {
    const opts = { rtol: 1e-13, atol: 1e-15 };
    for (const a of [0, 0.5, 0.9, 0.998]) {
      // Equatorial rays spanning capture, near-critical, and flyby regimes.
      for (const b of [3, 5, 5.19615, 6, 8, 12, 20]) {
        const res = integrateGeodesic(a, equatorialPhoton(a, 100, b, true), opts);
        expect(res.status === 'CAPTURED' || res.status === 'ESCAPED').toBe(true);
        expect(res.maxAbsH).toBeLessThan(DRIFT_BOUND);
        expect(res.maxDriftQ).toBeLessThan(DRIFT_BOUND);
      }
      // Inclined rays initialized through the ZAMO tetrad (exercises Q != 0).
      const m = createMetric();
      evalMetric(a, 100, Math.PI / 3, m);
      const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, a, 100));
      for (const n of [
        [-1, 0.03, 0.02],
        [-1, -0.05, 0.04],
        [-1, 0.06, -0.01],
      ] as Array<[number, number, number]>) {
        const y0 = photonFromTetrad(m, tetrad, n, { r: 100, theta: Math.PI / 3 });
        const res = integrateGeodesic(a, y0, opts);
        expect(res.status === 'CAPTURED' || res.status === 'ESCAPED').toBe(true);
        expect(res.maxAbsH).toBeLessThan(DRIFT_BOUND);
        expect(res.maxDriftQ).toBeLessThan(DRIFT_BOUND);
      }
    }
  });
});

describe('V-A3: Schwarzschild critical impact parameter', () => {
  it('capture/escape boundary bisects to b_c = 3*sqrt(3) within 1e-8', () => {
    const classify = (b: number): boolean =>
      integrateGeodesic(0, equatorialPhoton(0, 100, b, true), { rtol: 1e-12, atol: 1e-14 })
        .status === 'CAPTURED';
    let lo = 5.0; // captured
    let hi = 5.4; // escapes
    expect(classify(lo)).toBe(true);
    expect(classify(hi)).toBe(false);
    for (let i = 0; i < 45; i++) {
      const mid = 0.5 * (lo + hi);
      if (classify(mid)) lo = mid;
      else hi = mid;
    }
    expect(Math.abs(0.5 * (lo + hi) - B_CRIT_SCHWARZSCHILD)).toBeLessThan(1e-8);
  });
});

describe('V-A4: weak-field deflection', () => {
  it('matches alpha = 4/b + 15pi/(4b^2), residual bounded by the next series term 128/(3b^3)', () => {
    for (const b of [100, 300, 1000]) {
      // Start radius scales as b^2 so the finite-radius asymptote error
      // (~2b/r0) stays well below the O(b^-3) series term being verified.
      const r0 = b * b * 1e6;
      const res = integrateGeodesic(0, equatorialPhoton(0, r0, b, true), {
        rtol: 1e-13,
        atol: 1e-15,
        rEscape: r0,
      });
      expect(res.status).toBe('ESCAPED');
      const alpha = res.state[IPH]! - Math.PI;
      const expected = 4 / b + (15 * Math.PI) / (4 * b * b);
      // The measured deviation from the two-term series must itself be the
      // physical O(b^-3) term -- a sharper statement than a bare rel-tolerance.
      expect(Math.abs(alpha - expected)).toBeLessThan(1.5 * (128 / (3 * b ** 3)));
      expect(Math.abs(alpha - expected) / expected).toBeLessThan(1.1e-3);
    }
  });
});

describe('disk-plane crossing refinement', () => {
  it('Hermite-refined hit agrees with a tighter-tolerance re-integration to 1e-6', () => {
    const a = 0.9;
    const disk = { rInner: iscoRadius(a), rOuter: 20 };
    const m = createMetric();
    evalMetric(a, 100, Math.PI / 3, m);
    const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, a, 100));
    const y0 = photonFromTetrad(m, tetrad, [-1, 0.055, 0.02], { r: 100, theta: Math.PI / 3 });

    const loose = integrateGeodesic(a, Float64Array.from(y0), { rtol: 1e-11, atol: 1e-13, disk });
    const tight = integrateGeodesic(a, Float64Array.from(y0), { rtol: 1e-13, atol: 1e-15, disk });
    expect(loose.status).toBe('HIT_DISK');
    expect(tight.status).toBe('HIT_DISK');
    expect(Math.abs(loose.state[IR]! - tight.state[IR]!)).toBeLessThan(1e-6);
    expect(Math.abs(loose.state[IPH]! - tight.state[IPH]!)).toBeLessThan(1e-6);
  });
});
