/**
 * Novikov-Thorne disk LUT generator (IMPLEMENTATION.md M1, consumed by the
 * M4 shading pass). Stores T̂ = F̂^(1/4) (code-unit effective temperature per
 * unit Ṁ^(1/4)) rather than F̂: T̂ is what shading interpolates, is smooth,
 * and vanishes at the ISCO without a log singularity.
 *
 * Layout: row-major [aIndex][rIndex] over a uniform spin grid and a per-spin
 * log-radius grid u = ln(r/r_isco(a)) ∈ [0, ln(1e4)].
 */
import { iscoRadius } from '../src/physics/orbits';
import { fluxClosedForm } from '../src/physics/pageThorne';
import { writeLut } from './lutWriter';

const A_COUNT = 33;
const A_MAX = 0.998; // Thorne spin limit; also the renderer's cap (IMPLEMENTATION.md M3)
const R_COUNT = 512;
const U_MAX = Math.log(1e4);

const aValues: number[] = [];
const rIscoValues: number[] = [];
const data = new Float32Array(A_COUNT * R_COUNT);

for (let ia = 0; ia < A_COUNT; ia++) {
  const a = (ia / (A_COUNT - 1)) * A_MAX;
  const rIsco = iscoRadius(a);
  aValues.push(a);
  rIscoValues.push(rIsco);
  for (let ir = 0; ir < R_COUNT; ir++) {
    const u = (ir / (R_COUNT - 1)) * U_MAX;
    const r = rIsco * Math.exp(u);
    data[ia * R_COUNT + ir] = Math.pow(fluxClosedForm(a, r), 0.25);
  }
}

writeLut('assets/luts/nt-teff.bin', data, {
  description:
    'Novikov-Thorne effective temperature T̂ = F̂(r)^(1/4) per unit Ṁ^(1/4), Page & Thorne 1974 [R3]',
  layout: 'float32 rowMajor [aIndex][rIndex]',
  aCount: A_COUNT,
  rCount: R_COUNT,
  aValues,
  rIsco: rIscoValues,
  uGrid: `u = ln(r/r_isco(a)) uniform in [0, ${U_MAX}]`,
});
