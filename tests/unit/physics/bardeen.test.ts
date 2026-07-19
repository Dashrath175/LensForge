/**
 * V-A support for V-I1 (VALIDATION.md §1, §3): Bardeen's analytic Kerr shadow
 * family [R1] confronted with the float64 oracle, and the exact
 * conserved-quantity → camera-screen mapping used to overlay the contour.
 *
 * These closed forms and the geodesic integrator are independent derivations,
 * so agreement here validates the contour, the traced-photon sign
 * conventions, and the finite-distance ZAMO mapping all at once — the GPU
 * V-I1 comparison then only has to match the (already validated) overlay.
 */
import { describe, expect, it } from 'vitest';
import { integrateGeodesic } from '../../../src/physics/integrator';
import { createMetric, evalMetric } from '../../../src/physics/metric';
import { photonOrbitRadius } from '../../../src/physics/orbits';
import { equatorialPhoton, photonFromTetrad } from '../../../src/physics/photon';
import {
  bardeenContour,
  criticalImpactParameter,
  sphericalPhotonConserved,
} from '../../../src/physics/shadow';
import { orthonormalTetrad, zamoFourVelocity } from '../../../src/physics/tetrad';
import { directionForConserved } from '../../../src/validation/shadowContour';

const OPTS = { rtol: 1e-12, atol: 1e-14 };
const SPINS = [0.5, 0.9, 0.998];

describe('Bardeen family closed forms', () => {
  it('η vanishes at the equatorial photon orbits and ξ matches the critical impact parameter', () => {
    for (const a of SPINS) {
      for (const prograde of [true, false]) {
        const rPh = photonOrbitRadius(a, prograde);
        const { xi, eta } = sphericalPhotonConserved(a, rPh);
        expect(Math.abs(eta)).toBeLessThan(1e-9);
        expect(xi).toBeCloseTo(criticalImpactParameter(a, prograde), 10);
        // Prograde orbits carry positive L_z, retrograde negative.
        expect(xi > 0).toBe(prograde);
      }
    }
  });

  it('a → 0 limit recovers the Schwarzschild circle ξ² + η = 27 to O(a)', () => {
    // The residual is the physical O(a) shadow displacement (measured
    // 2.1e-5 at a = 1e-6), not a formula error — bound it at that scale.
    const a = 1e-6;
    for (const { xi, eta } of bardeenContour(a, 64)) {
      expect(Math.abs(xi * xi + eta - 27)).toBeLessThan(50 * a);
    }
  });
});

describe('critical impact parameter vs oracle capture bisection', () => {
  it('closed form matches the bisected capture/escape boundary to 1e-8 (both senses)', () => {
    for (const a of SPINS) {
      for (const prograde of [true, false]) {
        const bc = criticalImpactParameter(a, prograde);
        const captured = (b: number): boolean =>
          integrateGeodesic(a, equatorialPhoton(a, 100, b, true), OPTS).status === 'CAPTURED';
        // Capture side is |b| < |b_c| on each sense.
        let lo = bc - Math.sign(bc) * 0.2; // captured
        let hi = bc + Math.sign(bc) * 0.2; // escapes
        expect(captured(lo)).toBe(true);
        expect(captured(hi)).toBe(false);
        for (let i = 0; i < 45; i++) {
          const mid = 0.5 * (lo + hi);
          if (captured(mid)) lo = mid;
          else hi = mid;
        }
        expect(Math.abs(0.5 * (lo + hi) - bc)).toBeLessThan(1e-8);
      }
    }
  });
});

describe('shadow contour through the exact ZAMO screen mapping (finite camera distance)', () => {
  it('a = 0 mapping reproduces Synge’s exact angular radius', () => {
    const r0 = 1000;
    const m = createMetric();
    evalMetric(0, r0, Math.PI / 2, m);
    const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, 0, r0));
    const bc = 3 * Math.sqrt(3);
    const n = directionForConserved(m, tetrad, bc, 0);
    expect(n).not.toBeNull();
    const sinSynge = (bc * Math.sqrt(1 - 2 / r0)) / r0;
    expect(Math.abs(Math.hypot(n![1], n![2]) - sinSynge)).toBeLessThan(1e-14);
  });

  it('mapped Bardeen points bracket the oracle capture boundary (a ∈ {0.9, 0.998}, two inclinations)', () => {
    const r0 = 1000;
    for (const a of [0.9, 0.998]) {
      for (const theta0 of [Math.PI / 2, Math.PI / 3]) {
        const m = createMetric();
        evalMetric(a, r0, theta0, m);
        const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, a, r0));
        const c = Math.cos(theta0);
        const s = Math.sin(theta0);
        const contour = bardeenContour(a, 13);
        for (const { xi, eta } of contour) {
          const pth2 = eta + c * c * (a * a - (xi * xi) / (s * s));
          if (pth2 < 0) continue; // not visible at this inclination
          for (const sign of pth2 > 0 ? [1, -1] : [1]) {
            const n = directionForConserved(m, tetrad, xi, sign * Math.sqrt(pth2));
            if (!n) continue;
            // Scale the transverse angle in/out: shadow contours are convex and
            // contain the radial direction, so this crosses the boundary.
            for (const [scale, expected] of [
              [1 - 3e-4, 'CAPTURED'],
              [1 + 3e-4, 'ESCAPED'],
            ] as const) {
              const n2 = n[1] * scale;
              const n3 = n[2] * scale;
              const dir: [number, number, number] = [-Math.sqrt(1 - n2 * n2 - n3 * n3), n2, n3];
              const y0 = photonFromTetrad(m, tetrad, dir, { r: r0, theta: theta0 });
              expect(integrateGeodesic(a, y0, OPTS).status).toBe(expected);
            }
          }
        }
      }
    }
  });
});
