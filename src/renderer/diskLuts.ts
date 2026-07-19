/**
 * Novikov-Thorne and Planck LUT access (M4 shading; PHYSICS.md §7-§8).
 *
 * The tools/ generators own the LUT semantics (assets/luts/*.json); this
 * module loads the binaries and provides the pure slicing/sampling math that
 * the renderer (CPU side) and the V-O2/beaming validation harness share. The
 * WGSL samplers in shade.wgsl mirror `sampleNtSlice` / `samplePlanckXyz`
 * exactly, so a CPU-evaluated shade is the oracle for the GPU one.
 *
 * nt-teff:    T̂(a, u) = F̂^{1/4} per unit Ṁ^{1/4}, rows over a uniform spin
 *             grid, columns over u = ln(r/r_isco(a)) uniform in [0, uMax].
 * planck-xyz: blackbody XYZ (W m⁻² sr⁻¹) on a uniform log10 T grid.
 */

export interface NtLut {
  data: Float32Array<ArrayBuffer>;
  aValues: number[];
  aCount: number;
  rCount: number;
  uMax: number;
}

export interface PlanckLut {
  /** Flat [tIndex][X,Y,Z] triplets. */
  data: Float32Array<ArrayBuffer>;
  tCount: number;
  log10TMin: number;
  log10TMax: number;
}

export interface DiskLuts {
  nt: NtLut;
  planck: PlanckLut;
}

/**
 * Slice T̂(a, ·) at an arbitrary spin by linear interpolation between the two
 * bracketing spin rows. Rows share the same u grid (u is relative to each
 * row's own r_isco), so interpolating in u-space compares like radii and the
 * ISCO zero of T̂ is preserved exactly at u = 0.
 */
export function sliceNtAtSpin(nt: NtLut, a: number): Float32Array<ArrayBuffer> {
  const { aValues, rCount, data } = nt;
  const aClamped = Math.min(Math.max(a, aValues[0]!), aValues[aValues.length - 1]!);
  let hi = 1;
  while (hi < aValues.length - 1 && aValues[hi]! < aClamped) hi++;
  const lo = hi - 1;
  const t = (aClamped - aValues[lo]!) / (aValues[hi]! - aValues[lo]!);
  const slice = new Float32Array(rCount);
  for (let i = 0; i < rCount; i++) {
    slice[i] = (1 - t) * data[lo * rCount + i]! + t * data[hi * rCount + i]!;
  }
  return slice;
}

/** T̂ at BL radius r from a spin slice — CPU mirror of sampleNtTHat in shade.wgsl. */
export function sampleNtSlice(slice: Float32Array, uMax: number, rIsco: number, r: number): number {
  const u = Math.log(r / rIsco);
  const x = Math.min(Math.max(u / uMax, 0), 1) * (slice.length - 1);
  const i0 = Math.floor(x);
  const i1 = Math.min(i0 + 1, slice.length - 1);
  const f = x - i0;
  return (1 - f) * slice[i0]! + f * slice[i1]!;
}

/** Blackbody XYZ at temperature tK — CPU mirror of samplePlanckXyz in shade.wgsl. */
export function samplePlanckXyz(planck: PlanckLut, tK: number): [number, number, number] {
  const count = planck.tCount;
  const logT = Math.log10(Math.max(tK, 1e-6));
  const x =
    Math.min(Math.max((logT - planck.log10TMin) / (planck.log10TMax - planck.log10TMin), 0), 1) *
    (count - 1);
  const i0 = Math.floor(x);
  const i1 = Math.min(i0 + 1, count - 1);
  const f = x - i0;
  const d = planck.data;
  return [
    (1 - f) * d[3 * i0]! + f * d[3 * i1]!,
    (1 - f) * d[3 * i0 + 1]! + f * d[3 * i1 + 1]!,
    (1 - f) * d[3 * i0 + 2]! + f * d[3 * i1 + 2]!,
  ];
}

/** CIE XYZ (D65) → linear sRGB [R17] — CPU mirror of XYZ_TO_SRGB in shade.wgsl. */
export function xyzToLinearSrgb(x: number, y: number, z: number): [number, number, number] {
  return [
    Math.max(3.2406 * x - 1.5372 * y - 0.4986 * z, 0),
    Math.max(-0.9689 * x + 1.8758 * y + 0.0415 * z, 0),
    Math.max(0.0557 * x - 0.204 * y + 1.057 * z, 0),
  ];
}

/** Fetch both LUT binaries + metadata (Vite bundles them as hashed assets). */
export async function loadDiskLuts(): Promise<DiskLuts> {
  const load = async (
    bin: URL,
    json: URL,
  ): Promise<[Float32Array<ArrayBuffer>, Record<string, unknown>]> => {
    const [binRes, jsonRes] = await Promise.all([fetch(bin), fetch(json)]);
    if (!binRes.ok || !jsonRes.ok) throw new Error(`LUT fetch failed: ${bin}`);
    return [new Float32Array(await binRes.arrayBuffer()), (await jsonRes.json()) as Record<string, unknown>];
  };
  const [ntData, ntMeta] = await load(
    new URL('../../assets/luts/nt-teff.bin', import.meta.url),
    new URL('../../assets/luts/nt-teff.json', import.meta.url),
  );
  const [planckData, planckMeta] = await load(
    new URL('../../assets/luts/planck-xyz.bin', import.meta.url),
    new URL('../../assets/luts/planck-xyz.json', import.meta.url),
  );
  const uMatch = /\[0, ([0-9.]+)\]/.exec(String(ntMeta['uGrid']));
  const tMatch = /\[([0-9.]+), ([0-9.]+)\]/.exec(String(planckMeta['log10T']));
  if (!uMatch || !tMatch) throw new Error('LUT metadata missing grid ranges');
  return {
    nt: {
      data: ntData,
      aValues: ntMeta['aValues'] as number[],
      aCount: ntMeta['aCount'] as number,
      rCount: ntMeta['rCount'] as number,
      uMax: parseFloat(uMatch[1]!),
    },
    planck: {
      data: planckData,
      tCount: planckMeta['tCount'] as number,
      log10TMin: parseFloat(tMatch[1]!),
      log10TMax: parseFloat(tMatch[2]!),
    },
  };
}
