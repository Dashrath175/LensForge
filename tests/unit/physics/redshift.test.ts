/**
 * V-A7 (VALIDATION.md section 1): redshift factor limits and consistency.
 */
import { describe, expect, it } from 'vitest';
import { createMetric, evalMetric } from '../../../src/physics/metric';
import { equatorialPhoton } from '../../../src/physics/photon';
import { diskRedshiftCompact, redshiftToInfinity } from '../../../src/physics/redshift';
import { circularOrbitFourVelocity, staticFourVelocity } from '../../../src/physics/tetrad';

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
