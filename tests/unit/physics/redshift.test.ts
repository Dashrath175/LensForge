/**
 * V-A7 (VALIDATION.md section 1): redshift factor limits and consistency.
 */
import { describe, expect, it } from 'vitest';
import { IPPH, IPT, IR } from '../../../src/physics/hamiltonian';
import { integrateGeodesic } from '../../../src/physics/integrator';
import { createMetric, evalMetric } from '../../../src/physics/metric';
import { angularVelocity, circularUt, iscoRadius } from '../../../src/physics/orbits';
import { equatorialPhoton, photonFromTetrad } from '../../../src/physics/photon';
import { diskRedshiftCompact, redshiftToInfinity } from '../../../src/physics/redshift';
import {
  circularOrbitFourVelocity,
  orthonormalTetrad,
  staticFourVelocity,
  zamoFourVelocity,
} from '../../../src/physics/tetrad';

describe('V-A7: redshift factor', () => {
  it('static emitter -> pure gravitational redshift g = sqrt(1 - 2/r) (a = 0)', () => {
    const m = createMetric();
    for (const r of [3, 10, 100]) {
      evalMetric(0, r, Math.PI / 2, m);
      const u = staticFourVelocity(m, 0, r, Math.PI / 2);
      const photon = equatorialPhoton(0, r, 1.5, false);
      const g = redshiftToInfinity(photon, u);
      expect(g).toBeCloseTo(Math.sqrt(1 - 2 / r), 13);
    }
  });

  it('circular emitter, radially emitted photon -> g = sqrt(1 - 3/r) (a = 0, exact)', () => {
    // For b = 0 the Doppler term vanishes and gravitational + transverse
    // Doppler combine to the classic sqrt(1 - 3M/r).
    for (const r of [6, 8, 20, 100]) {
      const u = circularOrbitFourVelocity(0, r);
      const photon = equatorialPhoton(0, r, 0, false);
      const g = redshiftToInfinity(photon, u);
      expect(g).toBeCloseTo(Math.sqrt(1 - 3 / r), 13);
    }
  });

  it('compact Cunningham form agrees with the general contraction (a = 0.9)', () => {
    const a = 0.9;
    for (const r of [4, 8, 15]) {
      const u = circularOrbitFourVelocity(a, r);
      for (const b of [-2, 0, 1, 2]) {
        const photon = equatorialPhoton(a, r, b, false);
        const gGeneral = redshiftToInfinity(photon, u);
        const gCompact = diskRedshiftCompact(a, r, b);
        expect(Math.abs(gGeneral - gCompact) / gCompact).toBeLessThan(1e-12);
      }
    }
  });

  it('camera-tetrad g = 1/(u^t (E − Ω L)) equals the exact contraction on real disk hits (M4 kernel formula)', () => {
    // The WGSL diskRedshift (shaders/disk/kinematics.wgsl) evaluates
    // 1/(u^t (E − Ω L)) for tetrad-normalized camera rays, where
    // (−k·u)_obs = 1 by construction. Verify the algebra against the general
    // contraction on genuine HIT_DISK states from a ZAMO camera.
    for (const a of [0, 0.9]) {
      const disk = { rInner: iscoRadius(a), rOuter: 20 };
      const m = createMetric();
      evalMetric(a, 100, 1.309, m);
      const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, a, 100));
      let hits = 0;
      for (const n2 of [0.03, 0.05, 0.08, 0.12, 0.16]) {
        for (const n3 of [-0.05, 0, 0.05]) {
          const y0 = photonFromTetrad(m, tetrad, [-1, n2, n3], { r: 100, theta: 1.309 });
          const res = integrateGeodesic(a, y0, { rtol: 1e-13, atol: 1e-15, disk });
          if (res.status !== 'HIT_DISK') continue;
          hits++;
          const rHit = res.state[IR]!;
          const E = -y0[IPT]!;
          const L = y0[IPPH]!;
          const gKernel = 1 / (circularUt(a, rHit) * (E - angularVelocity(a, rHit) * L));
          // General contraction at the hit state; ν_obs uses (−k·u)_obs = 1.
          const u = circularOrbitFourVelocity(a, rHit);
          const gExact = redshiftToInfinity(res.state, u) / E;
          expect(Math.abs(gKernel - gExact) / gExact).toBeLessThan(1e-11);
        }
      }
      expect(hits).toBeGreaterThan(2);
    }
  });

  it('approaches special-relativistic Doppler at large radius: g(b = +/-L_max) brackets 1', () => {
    // At r = 1e6 the orbital speed is beta ~= 1e-3; a photon emitted along
    // (against) the motion is blueshifted (redshifted) by ~beta to leading order.
    const r = 1e6;
    const u = circularOrbitFourVelocity(0, r);
    const beta = Math.pow(r, -0.5); // Newtonian orbital speed, leading order
    // Forward/backward tangential photons at the emitter have b ~= +/- r (flat space).
    const gForward = redshiftToInfinity(equatorialPhoton(0, r, r * (1 - 1e-9), false), u);
    const gBackward = redshiftToInfinity(equatorialPhoton(0, r, -r * (1 - 1e-9), false), u);
    expect(gForward).toBeCloseTo(1 + beta, 4);
    expect(gBackward).toBeCloseTo(1 - beta, 4);
  });
});
