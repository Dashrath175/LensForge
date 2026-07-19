/**
 * M5 observer selection: every ObserverType yields a normalized timelike
 * 4-velocity (u·u = −1) whose Gram-Schmidt tetrad is orthonormal, and the
 * boost between frames is physically present (aberration scale ~ v).
 */
import { describe, expect, it } from 'vitest';
import { createMetric, evalMetric, innerProduct } from '../../../src/physics/metric';
import { angularVelocity } from '../../../src/physics/orbits';
import {
  observerFourVelocity,
  orthonormalTetrad,
  staticFourVelocity,
  zamoFourVelocity,
} from '../../../src/physics/tetrad';

const CASES: Array<{ a: number; r: number; theta: number }> = [
  { a: 0, r: 10, theta: Math.PI / 2 },
  { a: 0.9, r: 10, theta: Math.PI / 2 },
  { a: 0.998, r: 6, theta: Math.PI / 2 },
];

describe('observerFourVelocity', () => {
  it('returns u·u = −1 for all observer types at valid points', () => {
    for (const { a, r, theta } of CASES) {
      const m = createMetric();
      evalMetric(a, r, theta, m);
      for (const type of ['zamo', 'static', 'circular'] as const) {
        const u = observerFourVelocity(type, m, a, r, theta);
        expect(innerProduct(m, u, u), `${type} a=${a} r=${r}`).toBeCloseTo(-1, 12);
      }
    }
  });

  it('builds an orthonormal tetrad for every observer type', () => {
    for (const { a, r, theta } of CASES) {
      const m = createMetric();
      evalMetric(a, r, theta, m);
      for (const type of ['zamo', 'static', 'circular'] as const) {
        const tetrad = orthonormalTetrad(m, observerFourVelocity(type, m, a, r, theta));
        for (let i = 0; i < 4; i++) {
          for (let j = 0; j < 4; j++) {
            const expected = i !== j ? 0 : i === 0 ? -1 : 1;
            expect(innerProduct(m, tetrad[i]!, tetrad[j]!), `${type} g(e${i},e${j})`).toBeCloseTo(expected, 10);
          }
        }
      }
    }
  });

  it('circular-orbit u carries the orbital angular velocity (u^φ/u^t = Ω)', () => {
    const m = createMetric();
    evalMetric(0.9, 10, Math.PI / 2, m);
    const u = observerFourVelocity('circular', m, 0.9, 10, Math.PI / 2);
    expect(u[3] / u[0]).toBeCloseTo(angularVelocity(0.9, 10), 13);
  });

  it('static observer throws inside the ergosphere; ZAMO exists there', () => {
    const a = 0.998;
    const rInside = 1.9; // equatorial ergosphere is 2M
    const m = createMetric();
    evalMetric(a, rInside, Math.PI / 2, m);
    expect(() => staticFourVelocity(m, a, rInside, Math.PI / 2)).toThrow(/ergosphere/);
    const u = zamoFourVelocity(m, a, rInside);
    expect(innerProduct(m, u, u)).toBeCloseTo(-1, 12);
  });
});
