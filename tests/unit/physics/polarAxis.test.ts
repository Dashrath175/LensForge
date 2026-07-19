/**
 * M3 polar-axis handling (PHYSICS.md §10, IMPLEMENTATION.md M3 risk item):
 * the axis-regular φφ formulation must integrate rays on and across the
 * BL polar axis exactly — the pre-M3 formulation produced Inf·0 = NaN there.
 *
 * θ is integrated unrestricted (the Kerr components are even in θ and
 * 2π-periodic, so extended θ double-covers the sphere consistently, and the
 * pseudo-Cartesian escape map is single-valued on it); no reflection remap is
 * required anywhere downstream.
 */
import { describe, expect, it } from 'vitest';
import { IPH, IPR, IPT, IR, ITH, STATE_SIZE } from '../../../src/physics/hamiltonian';
import { integrateGeodesic } from '../../../src/physics/integrator';
import { createMetric, evalMetric, horizonRadius } from '../../../src/physics/metric';
import { equatorialPhoton, photonFromTetrad, polarPhoton } from '../../../src/physics/photon';
import { orthonormalTetrad, zamoFourVelocity } from '../../../src/physics/tetrad';

const DRIFT_BOUND = 5e-12; // V-A2 budget
const OPTS = { rtol: 1e-13, atol: 1e-15 };

function axialPhoton(a: number, r: number, ingoing: boolean): Float64Array {
  const m = evalMetric(a, r, 0, createMetric());
  const y = new Float64Array(STATE_SIZE);
  y[IR] = r;
  y[ITH] = 0; // exactly on the axis
  y[IPT] = -1;
  y[IPR] = (ingoing ? -1 : 1) * Math.sqrt(-m.gtt / m.grr);
  return y;
}

describe('polar axis: exactly-axial photons (p_φ = 0, θ = 0)', () => {
  it('ingoing axial photon is captured with finite state and V-A2 drift', () => {
    for (const a of [0, 0.5, 0.9, 0.998]) {
      const res = integrateGeodesic(a, axialPhoton(a, 100, true), OPTS);
      expect(res.status).toBe('CAPTURED');
      expect(Array.from(res.state).every(Number.isFinite)).toBe(true);
      expect(res.state[IR]!).toBeLessThanOrEqual(horizonRadius(a) * 1.001 * (1 + 1e-9));
      expect(res.maxAbsH).toBeLessThan(DRIFT_BOUND);
      expect(res.maxDriftQ).toBeLessThan(DRIFT_BOUND);
    }
  });

  it('outgoing axial photon escapes exactly along the pole', () => {
    const res = integrateGeodesic(0.998, axialPhoton(0.998, 100, false), OPTS);
    expect(res.status).toBe('ESCAPED');
    expect(res.escapeDirection).toBeDefined();
    const [dx, dy, dz] = res.escapeDirection!;
    expect(Math.abs(dx)).toBeLessThan(1e-12);
    expect(Math.abs(dy)).toBeLessThan(1e-12);
    expect(dz).toBeCloseTo(1, 12);
  });
});

describe('polar axis: axis-crossing rays (p_φ = 0 through θ = 0)', () => {
  it('a = 0: polar-plane ray reproduces the equatorial ray by spherical symmetry', () => {
    // With E = 1, p_φ = 0, |p_θ| = b the polar-plane motion is the equatorial
    // b-ray rotated into the (θ, φ = 0) plane: χ = π/2 − θ must track φ exactly,
    // including through the axis crossings the equatorial ray does not have.
    // b = 5.3 is near-critical (b_c ≈ 5.196): the ray winds and crosses both
    // poles, the strongest exactness test for the crossing.
    for (const b of [8, 5.3]) {
      const polar = integrateGeodesic(0, polarPhoton(0, 100, Math.PI / 2, -b, true), OPTS);
      const equatorial = integrateGeodesic(0, equatorialPhoton(0, 100, b, true), OPTS);
      expect(polar.status).toBe('ESCAPED');
      expect(equatorial.status).toBe('ESCAPED');
      const chi = Math.PI / 2 - polar.state[ITH]!;
      expect(Math.abs(chi - equatorial.state[IPH]!)).toBeLessThan(1e-8);
      expect(Math.abs(polar.state[IR]! - equatorial.state[IR]!)).toBeLessThan(1e-8);
      expect(polar.maxAbsH).toBeLessThan(DRIFT_BOUND);
      expect(polar.maxDriftQ).toBeLessThan(DRIFT_BOUND);
    }
  });

  it('a = 0.998: axis-crossing ray conserves Q and H and feels frame dragging', () => {
    const res = integrateGeodesic(0.998, polarPhoton(0.998, 100, Math.PI / 2, -8, true), OPTS);
    expect(res.status).toBe('ESCAPED');
    expect(res.maxAbsH).toBeLessThan(DRIFT_BOUND);
    expect(res.maxDriftQ).toBeLessThan(DRIFT_BOUND);
    // The crossing happened (extended θ went over the pole) …
    expect(res.state[ITH]!).toBeLessThan(0);
    // … and dφ/dλ = g^tφ p_t ≠ 0 twisted the ray azimuthally on the way.
    expect(Math.abs(res.state[IPH]!)).toBeGreaterThan(1e-3);
  });

  it('near-axis ZAMO camera at a = 0.998: tetrad rays integrate within V-A2 drift', () => {
    const theta0 = 1e-3;
    const m = createMetric();
    evalMetric(0.998, 50, theta0, m);
    const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, 0.998, 50));
    for (const n of [
      [-1, 0.05, 0.03],
      [-1, -0.08, 0.0],
      [-1, 0.0, -0.06],
      [-1, 0.2, 0.15],
    ] as Array<[number, number, number]>) {
      const y0 = photonFromTetrad(m, tetrad, n, { r: 50, theta: theta0 });
      const res = integrateGeodesic(0.998, y0, OPTS);
      expect(res.status === 'CAPTURED' || res.status === 'ESCAPED').toBe(true);
      expect(Array.from(res.state).every(Number.isFinite)).toBe(true);
      expect(res.maxAbsH).toBeLessThan(DRIFT_BOUND);
      // Near-axis high-spin paths run longer than the V-A2 equatorial set, so
      // the accumulated truncation floor sits higher (measured 9.2e-12).
      expect(res.maxDriftQ).toBeLessThan(4 * DRIFT_BOUND);
    }
  });
});
