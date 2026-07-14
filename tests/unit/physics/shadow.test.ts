/**
 * M1 exit-criterion smoke test (IMPLEMENTATION.md M1): end-to-end CPU shadow
 * render through the full camera path (tetrad -> photon -> integrator), with
 * the shadow's apparent angular radius checked against Synge's exact formula
 * sin(alpha_sh) = b_c sqrt(1 - 2/r0) / r0 for a static observer at r0 (a = 0).
 */
import { describe, expect, it } from 'vitest';
import { integrateGeodesic } from '../../../src/physics/integrator';
import { createMetric, evalMetric } from '../../../src/physics/metric';
import { photonFromTetrad } from '../../../src/physics/photon';
import { orthonormalTetrad, staticFourVelocity, type Tetrad } from '../../../src/physics/tetrad';
import type { MetricBL } from '../../../src/physics/metric';

const R_CAMERA = 1000;
const B_CRIT = 3 * Math.sqrt(3);
const ALPHA_SHADOW = Math.asin((B_CRIT * Math.sqrt(1 - 2 / R_CAMERA)) / R_CAMERA);

function cameraFrame(): { m: MetricBL; tetrad: Tetrad } {
  const m = createMetric();
  evalMetric(0, R_CAMERA, Math.PI / 2, m);
  const tetrad = orthonormalTetrad(m, staticFourVelocity(m, 0, R_CAMERA, Math.PI / 2));
  return { m, tetrad };
}

/** True if the pinhole ray at angles (ax, ay) from the inward optical axis is captured. */
function rayCaptured(
  m: MetricBL,
  tetrad: Tetrad,
  ax: number,
  ay: number,
  rtol: number,
): boolean {
  // Optical axis is -e1 (inward radial); e2/e3 span the image plane.
  const n: [number, number, number] = [-1, Math.tan(ax), Math.tan(ay)];
  const y0 = photonFromTetrad(m, tetrad, n, { r: R_CAMERA, theta: Math.PI / 2 });
  return integrateGeodesic(0, y0, { rtol, atol: rtol * 0.01 }).status === 'CAPTURED';
}

describe('M1 smoke: Schwarzschild shadow from the camera path', () => {
  it('shadow angular radius matches Synge to 1e-8 (bisection along a scanline)', () => {
    const { m, tetrad } = cameraFrame();
    let lo = ALPHA_SHADOW * 0.8; // captured
    let hi = ALPHA_SHADOW * 1.2; // escapes
    expect(rayCaptured(m, tetrad, lo, 0, 1e-12)).toBe(true);
    expect(rayCaptured(m, tetrad, hi, 0, 1e-12)).toBe(false);
    for (let i = 0; i < 40; i++) {
      const mid = 0.5 * (lo + hi);
      if (rayCaptured(m, tetrad, mid, 0, 1e-12)) lo = mid;
      else hi = mid;
    }
    expect(Math.abs(0.5 * (lo + hi) - ALPHA_SHADOW)).toBeLessThan(1e-8);
  });

  it('renders a 33x33 image whose captured-pixel count matches the shadow disk area', () => {
    const { m, tetrad } = cameraFrame();
    const N = 33;
    const halfFov = 2.2 * ALPHA_SHADOW;
    let captured = 0;
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const ax = ((2 * (i + 0.5)) / N - 1) * halfFov;
        const ay = ((2 * (j + 0.5)) / N - 1) * halfFov;
        if (rayCaptured(m, tetrad, ax, ay, 1e-8)) captured++;
      }
    }
    const expected = (N * N * Math.PI * ALPHA_SHADOW ** 2) / (2 * halfFov) ** 2;
    expect(Math.abs(captured - expected) / expected).toBeLessThan(0.15);
  });
});
