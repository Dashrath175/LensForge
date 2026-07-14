/**
 * V-I4 (VALIDATION.md section 3), oracle-side measurement for M2: the Einstein
 * ring angle for a source at infinity directly behind the lens must match the
 * weak-field lens equation theta_E = 2/sqrt(r0) (D_S -> infinity limit).
 *
 * Camera far out (r0 = 1e6) so higher-order deflection and finite-distance
 * corrections (~0.4% combined) sit inside the 1% criterion.
 */
import { describe, expect, it } from 'vitest';
import { integrateGeodesic } from '../../../src/physics/integrator';
import { createMetric, evalMetric } from '../../../src/physics/metric';
import { photonFromTetrad } from '../../../src/physics/photon';
import { orthonormalTetrad, staticFourVelocity } from '../../../src/physics/tetrad';

const R_CAMERA = 1e6;
const THETA_E_PREDICTED = 2 / Math.sqrt(R_CAMERA);

describe('V-I4: Einstein ring angle (oracle)', () => {
  it('ring angle matches the weak-field lens equation to < 1%', () => {
    const m = createMetric();
    evalMetric(0, R_CAMERA, Math.PI / 2, m);
    const tetrad = orthonormalTetrad(m, staticFourVelocity(m, 0, R_CAMERA, Math.PI / 2));

    // Escape-direction y-component as a function of launch angle from the
    // inward optical axis (equatorial plane, +phi side). Underbent rays keep
    // dir_y > 0; overbent rays cross the axis (dir_y < 0); the root is the ring.
    const escapeDirY = (angle: number): number => {
      const y0 = photonFromTetrad(m, tetrad, [-Math.cos(angle), 0, Math.sin(angle)], {
        r: R_CAMERA,
        theta: Math.PI / 2,
      });
      const res = integrateGeodesic(0, y0, { rtol: 1e-12, atol: 1e-14, rEscape: R_CAMERA });
      expect(res.status).toBe('ESCAPED');
      return res.escapeDirection![1];
    };

    let lo = THETA_E_PREDICTED * 0.9; // overbent
    let hi = THETA_E_PREDICTED * 1.1; // underbent
    expect(escapeDirY(lo)).toBeLessThan(0);
    expect(escapeDirY(hi)).toBeGreaterThan(0);
    for (let i = 0; i < 30; i++) {
      const mid = 0.5 * (lo + hi);
      if (escapeDirY(mid) < 0) lo = mid;
      else hi = mid;
    }
    const thetaMeasured = 0.5 * (lo + hi);
    expect(Math.abs(thetaMeasured - THETA_E_PREDICTED) / THETA_E_PREDICTED).toBeLessThan(0.01);
  });
});
