/**
 * V-I5 metric self-test (M8): the SSIM implementation used by the golden
 * comparison, checked against closed-form values and known invariances.
 */
import { describe, expect, it } from 'vitest';
import { computeSsim, rgba8ToLuma } from '../../../src/validation/ssim';

const W = 64;
const H = 48;

function image(fn: (x: number, y: number) => number): Float64Array {
  const img = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) img[y * W + x] = fn(x, y);
  return img;
}

/** Deterministic pseudo-noise (no RNG dependence in tests). */
function hashNoise(x: number, y: number): number {
  const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return s - Math.floor(s);
}

describe('V-I5 SSIM metric (M8)', () => {
  it('identical images score exactly 1', () => {
    const a = image((x, y) => 128 + 80 * Math.sin(x / 5) * Math.cos(y / 7));
    const r = computeSsim(a, a, W, H);
    expect(r.mean).toBeCloseTo(1, 12);
    expect(r.min).toBeCloseTo(1, 12);
  });

  it('matches the closed form for constant images', () => {
    // For constant images the variance/covariance term is C2/C2 = 1 and
    // SSIM = (2 μa μb + C1) / (μa² + μb² + C1) with C1 = (0.01·255)².
    const a = image(() => 100);
    const b = image(() => 110);
    const C1 = (0.01 * 255) ** 2;
    const expected = (2 * 100 * 110 + C1) / (100 * 100 + 110 * 110 + C1);
    const r = computeSsim(a, b, W, H);
    expect(r.mean).toBeCloseTo(expected, 12);
    expect(r.min).toBeCloseTo(expected, 12);
  });

  it('is symmetric and penalizes noise monotonically', () => {
    const base = image((x, y) => 128 + 60 * Math.sin(x / 4 + y / 9));
    const noisy = (amp: number): Float64Array =>
      image((x, y) => {
        const v = base[y * W + x]! + amp * (hashNoise(x, y) - 0.5);
        return Math.min(255, Math.max(0, v));
      });
    const weak = computeSsim(base, noisy(10), W, H);
    const strong = computeSsim(base, noisy(60), W, H);
    expect(weak.mean).toBeLessThan(1);
    expect(strong.mean).toBeLessThan(weak.mean);
    const sym = computeSsim(noisy(10), base, W, H);
    expect(sym.mean).toBeCloseTo(weak.mean, 12);
  });

  it('detects structural change far more strongly than uniform brightness', () => {
    const a = image((x, y) => 128 + 80 * Math.sin(x / 5) * Math.cos(y / 6));
    const brightened = image((x, y) => Math.min(255, a[y * W + x]! + 5));
    const shifted = image((x, y) => a[y * W + ((x + 8) % W)]!);
    const rBright = computeSsim(a, brightened, W, H);
    const rShift = computeSsim(a, shifted, W, H);
    expect(rBright.mean).toBeGreaterThan(0.99);
    expect(rShift.mean).toBeLessThan(0.6);
  });

  it('localizes the worst window', () => {
    const a = image(() => 120);
    const b = Float64Array.from(a);
    // A local defect at (40, 20): the worst window must land on it.
    for (let y = 18; y <= 22; y++) for (let x = 38; x <= 42; x++) b[y * W + x] = 250;
    const r = computeSsim(a, b, W, H);
    expect(Math.abs(r.minX - 40)).toBeLessThanOrEqual(3);
    expect(Math.abs(r.minY - 20)).toBeLessThanOrEqual(3);
    expect(r.min).toBeLessThan(0.5);
  });

  it('rgba8ToLuma applies Rec. 709 weights', () => {
    const bytes = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 100, 100, 100, 255]);
    const luma = rgba8ToLuma(bytes, 4, 1);
    expect(luma[0]).toBeCloseTo(0.2126 * 255, 10);
    expect(luma[1]).toBeCloseTo(0.7152 * 255, 10);
    expect(luma[2]).toBeCloseTo(0.0722 * 255, 10);
    expect(luma[3]).toBeCloseTo(100, 10);
  });

  it('rejects mismatched dimensions', () => {
    const a = image(() => 1);
    expect(() => computeSsim(a, a.subarray(0, W * H - 1), W, H)).toThrow();
    expect(() => computeSsim(new Float64Array(25), new Float64Array(25), 5, 5)).toThrow();
  });
});
