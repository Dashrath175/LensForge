/**
 * Planck → CIE 1931 XYZ LUT generator (PHYSICS.md §8; consumed by the M4
 * shading pass, which evaluates it at T_obs = g·T_eff).
 *
 * X(T) = ∫ B_λ(λ,T) x̄(λ) dλ over [360, 830] nm (Y, Z analogous), stored on a
 * uniform log10(T) grid. Color-matching functions use the multi-lobe Gaussian
 * fits of Wyman, Sloan & Shirley [R22] (≈1% accuracy — ample for rendering;
 * exact tables are a drop-in replacement if ever needed).
 * Units: W · m⁻² · sr⁻¹ (spectral radiance integrated over wavelength).
 */
import { BOLTZMANN_CONSTANT, PLANCK_CONSTANT, SPEED_OF_LIGHT } from '../src/core/constants';
import { writeLut } from './lutWriter';

const T_COUNT = 512;
const LOG10_T_MIN = 2; // 100 K
const LOG10_T_MAX = 7; // 10^7 K
const LAMBDA_MIN_NM = 360;
const LAMBDA_MAX_NM = 830;
const LAMBDA_STEP_NM = 1;

/** Piecewise Gaussian with distinct left/right widths (Wyman et al. [R22]). */
function lobe(x: number, mu: number, sigmaL: number, sigmaR: number): number {
  const t = (x - mu) / (x < mu ? sigmaL : sigmaR);
  return Math.exp(-0.5 * t * t);
}

function cieX(nm: number): number {
  return (
    1.056 * lobe(nm, 599.8, 37.9, 31.0) +
    0.362 * lobe(nm, 442.0, 16.0, 26.7) -
    0.065 * lobe(nm, 501.1, 20.4, 26.2)
  );
}

function cieY(nm: number): number {
  return 0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1);
}

function cieZ(nm: number): number {
  return 1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8);
}

/** Spectral radiance B_λ(λ, T) in W · m⁻³ · sr⁻¹. */
function planck(lambdaM: number, tK: number): number {
  const h = PLANCK_CONSTANT;
  const c = SPEED_OF_LIGHT;
  const kB = BOLTZMANN_CONSTANT;
  const x = (h * c) / (lambdaM * kB * tK);
  if (x > 700) return 0; // exp overflow guard; radiance is zero to f64 anyway
  return (2 * h * c * c) / (lambdaM ** 5 * (Math.exp(x) - 1));
}

const data = new Float32Array(T_COUNT * 3);
for (let i = 0; i < T_COUNT; i++) {
  const logT = LOG10_T_MIN + (i / (T_COUNT - 1)) * (LOG10_T_MAX - LOG10_T_MIN);
  const T = Math.pow(10, logT);
  let x = 0;
  let y = 0;
  let z = 0;
  // Trapezoidal rule at 1 nm; CMF lobes are smooth on ~10 nm scales.
  for (let nm = LAMBDA_MIN_NM; nm <= LAMBDA_MAX_NM; nm += LAMBDA_STEP_NM) {
    const w = nm === LAMBDA_MIN_NM || nm === LAMBDA_MAX_NM ? 0.5 : 1;
    const b = planck(nm * 1e-9, T) * w;
    x += b * cieX(nm);
    y += b * cieY(nm);
    z += b * cieZ(nm);
  }
  const dLambda = LAMBDA_STEP_NM * 1e-9;
  data[i * 3] = x * dLambda;
  data[i * 3 + 1] = y * dLambda;
  data[i * 3 + 2] = z * dLambda;
}

writeLut('assets/luts/planck-xyz.bin', data, {
  description: 'Blackbody spectral radiance integrated against CIE 1931 XYZ CMFs (Wyman et al. fits [R22])',
  layout: 'float32 [tIndex][X,Y,Z]',
  tCount: T_COUNT,
  log10T: `uniform in [${LOG10_T_MIN}, ${LOG10_T_MAX}]`,
  units: 'W m^-2 sr^-1',
});
