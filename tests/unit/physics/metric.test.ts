/**
 * V-A1 (VALIDATION.md §1): metric and tetrad identities, plus verification of
 * the hand-derived inverse-metric gradients against Richardson finite
 * differences (guards the analytic derivation that the WGSL port will copy).
 */
import { describe, expect, it } from 'vitest';
import { richardsonDerivative } from '../../../src/core/numeric';
import {
  createGradients,
  createMetric,
  evalInverseMetricGradients,
  evalMetric,
  horizonRadius,
  innerProduct,
} from '../../../src/physics/metric';
import {
  circularOrbitFourVelocity,
  orthonormalTetrad,
  staticFourVelocity,
  zamoFourVelocity,
} from '../../../src/physics/tetrad';

const SPINS = [0, 0.5, 0.9, 0.998];
const THETAS = [1e-3, Math.PI / 4, Math.PI / 2, 2.0, Math.PI - 1e-3];

function sampleRadii(a: number): number[] {
  const rp = horizonRadius(a);
  return [rp * 1.001, rp * 1.1, 3, 10, 100, 1e4];
}

function covMatrix(m: ReturnType<typeof createMetric>): number[][] {
  return [
    [m.g_tt, 0, 0, m.g_tph],
    [0, m.g_rr, 0, 0],
    [0, 0, m.g_thth, 0],
    [m.g_tph, 0, 0, m.g_phph],
  ];
}

function conMatrix(m: ReturnType<typeof createMetric>): number[][] {
  return [
    [m.gtt, 0, 0, m.gtph],
    [0, m.grr, 0, 0],
    [0, 0, m.gthth, 0],
    [m.gtph, 0, 0, m.gphph],
  ];
}

describe('V-A1: metric identities', () => {
  it('g^{μν} g_{νσ} = δ^μ_σ over the (a, r, θ) grid including near-horizon and near-axis', () => {
    const m = createMetric();
    for (const a of SPINS) {
      const rp = horizonRadius(a);
      for (const r of sampleRadii(a)) {
        for (const th of THETAS) {
          // Near-horizon/near-axis points are intrinsically ill-conditioned
          // (Δ and sin²θ cancellations inside the components themselves);
          // measured: ≤ 8.4e-16 well-conditioned, ≤ 1.6e-12 at corners.
          const cornerPoint = r < rp * 1.05 || th < 0.01 || th > Math.PI - 0.01;
          const tol = cornerPoint ? 1e-11 : 1e-12;
          evalMetric(a, r, th, m);
          const G = conMatrix(m);
          const g = covMatrix(m);
          for (let mu = 0; mu < 4; mu++) {
            for (let sg = 0; sg < 4; sg++) {
              let sum = 0;
              let scale = 0;
              for (let nu = 0; nu < 4; nu++) {
                sum += G[mu]![nu]! * g[nu]![sg]!;
                scale += Math.abs(G[mu]![nu]! * g[nu]![sg]!);
              }
              const target = mu === sg ? 1 : 0;
              expect(Math.abs(sum - target)).toBeLessThanOrEqual(tol * Math.max(1, scale));
            }
          }
        }
      }
    }
  });

  it('analytic inverse-metric gradients match Richardson finite differences', () => {
    const m = createMetric();
    const grad = createGradients();
    const keys = ['gtt', 'gtph', 'grr', 'gthth', 'gphph'] as const;
    for (const a of SPINS) {
      for (const r of [horizonRadius(a) * 1.05, 3, 10, 200]) {
        for (const th of [0.3, Math.PI / 2, 2.5]) {
          evalInverseMetricGradients(a, r, th, grad);
          for (const key of keys) {
            const fdR = richardsonDerivative((x) => evalMetric(a, x, th, m)[key], r, 1e-5 * r);
            const fdTh = richardsonDerivative((x) => evalMetric(a, r, x, m)[key], th, 1e-5);
            const anR = grad[`dr_${key}` as keyof typeof grad];
            const anTh = grad[`dth_${key}` as keyof typeof grad];
            expect(Math.abs(anR - fdR)).toBeLessThanOrEqual(1e-6 * (Math.abs(anR) + 1));
            expect(Math.abs(anTh - fdTh)).toBeLessThanOrEqual(1e-6 * (Math.abs(anTh) + 1));
          }
        }
      }
    }
  });

  it('tetrads satisfy g(e_a, e_b) = η_ab for ZAMO, static, and circular observers', () => {
    const m = createMetric();
    const eta = [-1, 1, 1, 1];
    const cases: Array<{ a: number; r: number; th: number; kind: string }> = [];
    for (const a of SPINS) {
      cases.push({ a, r: horizonRadius(a) * 1.01, th: Math.PI / 2, kind: 'zamo' });
      cases.push({ a, r: 5, th: 1.0, kind: 'zamo' });
      cases.push({ a, r: 10, th: Math.PI / 2, kind: 'static' }); // outside ergosphere
      cases.push({ a, r: 10, th: Math.PI / 2, kind: 'circular' });
    }
    for (const { a, r, th, kind } of cases) {
      evalMetric(a, r, th, m);
      const u =
        kind === 'zamo'
          ? zamoFourVelocity(m, a, r)
          : kind === 'static'
            ? staticFourVelocity(m, a, r, th)
            : circularOrbitFourVelocity(a, r);
      const tetrad = orthonormalTetrad(m, u);
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          const target = i === j ? eta[i]! : 0;
          expect(Math.abs(innerProduct(m, tetrad[i]!, tetrad[j]!) - target)).toBeLessThanOrEqual(1e-12);
        }
      }
    }
  });

  it('static observer construction throws inside the ergosphere', () => {
    const m = createMetric();
    evalMetric(0.9, 1.5, Math.PI / 2, m); // r < r_E(π/2) = 2
    expect(() => staticFourVelocity(m, 0.9, 1.5, Math.PI / 2)).toThrow(RangeError);
  });
});
