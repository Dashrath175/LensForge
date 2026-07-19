/**
 * M4 LUT access math (src/renderer/diskLuts.ts): the pure slicing/sampling
 * functions that the CPU shade oracle (V-I3 beaming check) shares with the
 * renderer, and that shade.wgsl mirrors. Synthetic LUTs — the physical
 * content of the real LUTs is validated by V-A6 (tools regenerate them from
 * the tested closed form).
 */
import { describe, expect, it } from 'vitest';
import {
  sampleNtSlice,
  samplePlanckXyz,
  sliceNtAtSpin,
  xyzToLinearSrgb,
  type NtLut,
  type PlanckLut,
} from '../../../src/renderer/diskLuts';

function syntheticNt(): NtLut {
  // Three spin rows with recognizable values: row k = k + i/10.
  const rCount = 8;
  const aValues = [0, 0.5, 1];
  const data = new Float32Array(3 * rCount);
  for (let k = 0; k < 3; k++) for (let i = 0; i < rCount; i++) data[k * rCount + i] = k + i / 10;
  return { data, aValues, aCount: 3, rCount, uMax: Math.log(100) };
}

describe('M4: NT LUT slicing and sampling', () => {
  it('slice at a grid spin reproduces that row exactly', () => {
    const nt = syntheticNt();
    const slice = sliceNtAtSpin(nt, 0.5);
    for (let i = 0; i < nt.rCount; i++) expect(slice[i]).toBeCloseTo(1 + i / 10, 6);
  });

  it('slice between grid spins interpolates linearly', () => {
    const nt = syntheticNt();
    const slice = sliceNtAtSpin(nt, 0.25);
    for (let i = 0; i < nt.rCount; i++) expect(slice[i]).toBeCloseTo(0.5 + i / 10, 6);
  });

  it('radius sampling: exact at grid nodes, clamped at the edges', () => {
    const nt = syntheticNt();
    const slice = sliceNtAtSpin(nt, 0);
    const rIsco = 6;
    // Node i sits at r = rIsco * exp(uMax * i/(rCount-1)).
    for (const i of [0, 3, 7]) {
      const r = rIsco * Math.exp((nt.uMax * i) / (nt.rCount - 1));
      expect(sampleNtSlice(slice, nt.uMax, rIsco, r)).toBeCloseTo(i / 10, 5);
    }
    expect(sampleNtSlice(slice, nt.uMax, rIsco, rIsco * 0.5)).toBeCloseTo(0, 6); // below ISCO clamps to the edge zero
    expect(sampleNtSlice(slice, nt.uMax, rIsco, rIsco * 1e5)).toBeCloseTo(0.7, 6);
  });
});

describe('M4: Planck LUT sampling and color conversion', () => {
  it('exact at grid temperatures, interpolates between', () => {
    const tCount = 6;
    const planck: PlanckLut = {
      data: Float32Array.from({ length: tCount * 3 }, (_, i) => i),
      tCount,
      log10TMin: 2,
      log10TMax: 7,
    };
    // Node i is at log10 T = 2 + i.
    const [x1, y1, z1] = samplePlanckXyz(planck, 1e3);
    expect(x1).toBeCloseTo(3, 5);
    expect(y1).toBeCloseTo(4, 5);
    expect(z1).toBeCloseTo(5, 5);
    // Halfway in log10 T between nodes 1 and 2.
    const [x15] = samplePlanckXyz(planck, Math.pow(10, 3.5));
    expect(x15).toBeCloseTo(4.5, 5);
    // Clamped below/above the domain.
    expect(samplePlanckXyz(planck, 1)[0]).toBeCloseTo(0, 6);
    expect(samplePlanckXyz(planck, 1e9)[0]).toBeCloseTo(15, 5);
  });

  it('XYZ→linear sRGB maps D65 white to ~(1,1,1) and clamps out-of-gamut to 0', () => {
    const [r, g, b] = xyzToLinearSrgb(0.9505, 1.0, 1.089);
    expect(r).toBeCloseTo(1, 2);
    expect(g).toBeCloseTo(1, 2);
    expect(b).toBeCloseTo(1, 2);
    // A saturated spectral-blue XYZ goes negative in R before clamping.
    expect(xyzToLinearSrgb(0.1, 0.05, 0.5)[0]).toBeGreaterThanOrEqual(0);
  });
});
