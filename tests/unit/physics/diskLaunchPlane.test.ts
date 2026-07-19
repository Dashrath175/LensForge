/**
 * Disk-crossing transversality (M5 finding): a camera placed exactly ON the
 * equatorial plane (θ = π/2) inside the disk band must not self-intersect
 * the razor-thin disk at its own launch point — cos(θ₀) sits at ULP scale
 * with an arbitrary sign there, so without the plane-launch quantization
 * (PLANE_LAUNCH_EPS) every equatorial observer with r < r_out saw 100%
 * HIT_DISK at σ = 0. Genuine crossings ahead of the camera must survive.
 */
import { describe, expect, it } from 'vitest';
import { integrateGeodesic, PLANE_LAUNCH_EPS } from '../../../src/physics/integrator';
import { createMetric, evalMetric } from '../../../src/physics/metric';
import { photonFromTetrad } from '../../../src/physics/photon';
import { orthonormalTetrad, zamoFourVelocity } from '../../../src/physics/tetrad';
import { iscoRadius } from '../../../src/physics/orbits';

const A = 0.9;
const DISK = { rInner: iscoRadius(A), rOuter: 20 };
const OPTS = { rtol: 1e-12, atol: 1e-14, disk: DISK };

function rayFrom(r: number, theta: number, n: [number, number, number]): Float64Array {
  const m = createMetric();
  evalMetric(A, r, theta, m);
  const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, A, r));
  return photonFromTetrad(m, tetrad, n, { r, theta });
}

describe('rays launched exactly on the equatorial plane inside the disk band', () => {
  it('never hit the disk at their own launch point (r = 12 ∈ band)', () => {
    // Expected outcomes verified against the f64 oracle: upward-tilted ray
    // escapes, downward-tilted inward ray plunges/hits far from the camera.
    const cases: Array<{ n: [number, number, number]; forbidRHit: number }> = [
      { n: [-1, 0.3, 0], forbidRHit: 12 },
      { n: [-1, -0.3, 0.2], forbidRHit: 12 },
      { n: [1, 0.5, -0.4], forbidRHit: 12 },
    ];
    for (const { n, forbidRHit } of cases) {
      const res = integrateGeodesic(A, rayFrom(12, Math.PI / 2, n), OPTS);
      if (res.status === 'HIT_DISK') {
        expect(Math.abs(res.state[1]! - forbidRHit), `n=${n.join(',')}`).toBeGreaterThan(1e-3);
        expect(res.steps).toBeGreaterThan(1);
      } else {
        expect(['CAPTURED', 'ESCAPED'], `n=${n.join(',')}`).toContain(res.status);
      }
      expect(res.steps).toBeGreaterThan(1);
    }
  });

  it('quantizes the launch θ onto the departure side within the snap band', () => {
    // A launch at π/2 + ε/2 (inside the band) moving toward larger θ starts
    // at π/2 + ε exactly; the same start moving toward smaller θ starts at
    // π/2 − ε. Observable: no immediate hit either way.
    for (const sgn of [1, -1]) {
      const res = integrateGeodesic(
        A,
        rayFrom(12, Math.PI / 2 + (sgn * PLANE_LAUNCH_EPS) / 2, [-1, sgn * 0.4, 0]),
        OPTS,
      );
      expect(res.steps).toBeGreaterThan(1);
      if (res.status === 'HIT_DISK') {
        expect(Math.abs(res.state[1]! - 12)).toBeGreaterThan(1e-3);
      }
    }
  });

  it('still records genuine crossings from off-plane launches (M4 behavior intact)', () => {
    // Verified against the oracle: hits at r ≈ 17.43 / 14.67 in-band.
    const expected: Array<{ r: number; theta: number; n: [number, number, number]; rHit: number }> = [
      { r: 30, theta: 1.2, n: [-1, 0.45, 0], rHit: 17.425 },
      { r: 25, theta: 1.3, n: [-1, 0.35, 0], rHit: 14.666 },
    ];
    for (const c of expected) {
      const res = integrateGeodesic(A, rayFrom(c.r, c.theta, c.n), OPTS);
      expect(res.status, `r=${c.r}`).toBe('HIT_DISK');
      expect(res.state[1]!).toBeCloseTo(c.rHit, 2);
    }
  });
});
