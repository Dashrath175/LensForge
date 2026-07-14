/**
 * PHYSICS.md §10 — adaptive Dormand-Prince RK5(4) geodesic integrator [R12]
 * with PI step control [R11], conserved-quantity monitors, horizon/plane
 * step clamps, and Hermite-refined equatorial disk crossings.
 *
 * This float64 implementation is the project oracle (MASTER_PLAN.md §1.6);
 * the M2 WGSL kernel ports this file's exact formulation.
 */
import {
  carterConstant,
  createRhsScratch,
  geodesicRhs,
  hamiltonianResidual,
  IPH,
  IR,
  ITH,
  STATE_SIZE,
} from './hamiltonian';
import { cartesianDirection, createMetric, evalMetric, horizonRadius } from './metric';

// Dormand-Prince RK5(4) tableau [R12].
const A21 = 1 / 5;
const A31 = 3 / 40, A32 = 9 / 40;
const A41 = 44 / 45, A42 = -56 / 15, A43 = 32 / 9;
const A51 = 19372 / 6561, A52 = -25360 / 2187, A53 = 64448 / 6561, A54 = -212 / 729;
const A61 = 9017 / 3168, A62 = -355 / 33, A63 = 46732 / 5247, A64 = 49 / 176, A65 = -5103 / 18656;
// 5th-order solution weights (also row 7 → FSAL).
const B1 = 35 / 384, B3 = 500 / 1113, B4 = 125 / 192, B5 = -2187 / 6784, B6 = 11 / 84;
// Error weights: b − b̂ (5th minus embedded 4th order).
const E1 = B1 - 5179 / 57600;
const E3 = B3 - 7571 / 16695;
const E4 = B4 - 393 / 640;
const E5 = B5 - -92097 / 339200;
const E6 = B6 - 187 / 2100;
const E7 = -1 / 40;

export type RayStatus = 'CAPTURED' | 'ESCAPED' | 'HIT_DISK' | 'MAX_STEPS';

export interface DiskGeometry {
  rInner: number;
  rOuter: number;
}

export interface GeodesicOptions {
  rtol?: number;
  atol?: number;
  maxSteps?: number;
  /** Escape radius; PHYSICS.md §10 default 1e4 (tests override for weak-field work). */
  rEscape?: number;
  /** Capture at r ≤ r₊(1 + ε). */
  horizonEpsilon?: number;
  /** Equatorial thin-disk band; crossings inside it terminate the ray. */
  disk?: DiskGeometry | null;
  /** Per-step relative monitor budget (Q drift, |H|); breaching steps are rejected once. */
  monitorTolerance?: number;
}

export interface GeodesicResult {
  status: RayStatus;
  /** Final state: at capture/escape/step-limit the last accepted state; at a disk hit, the refined crossing state. */
  state: Float64Array;
  lambda: number;
  steps: number;
  rejectedSteps: number;
  /** Max |H| normalized by the magnitude sum of its terms, and max relative Carter-constant drift. */
  maxAbsH: number;
  maxDriftQ: number;
  /** Asymptotic sky direction (unit pseudo-Cartesian), set when status is ESCAPED. */
  escapeDirection?: [number, number, number];
}

const DEFAULTS = {
  rtol: 1e-12,
  atol: 1e-14,
  maxSteps: 1_000_000,
  rEscape: 1e4,
  horizonEpsilon: 1e-3,
  monitorTolerance: 1e-8,
};

const SAFETY = 0.9;
const MIN_SCALE = 0.2;
const MAX_SCALE = 5.0;
const PI_ALPHA = 0.7 / 5;
const PI_BETA = 0.4 / 5;
/** Max |Δθ| per step while inside the disk's radial band, so plane crossings cannot be skipped. */
const THETA_STEP_CAP = 0.15;

/** Cubic Hermite interpolation of the full state across an accepted step, σ ∈ [0, 1]. */
function hermiteState(
  y0: Float64Array,
  f0: Float64Array,
  y1: Float64Array,
  f1: Float64Array,
  h: number,
  sigma: number,
  out: Float64Array,
): void {
  const s2 = sigma * sigma;
  const s3 = s2 * sigma;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + sigma;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;
  for (let i = 0; i < STATE_SIZE; i++) {
    out[i] = h00 * y0[i]! + h10 * h * f0[i]! + h01 * y1[i]! + h11 * h * f1[i]!;
  }
}

/** cosθ of the Hermite-interpolated state at σ (plane-crossing indicator function). */
function hermiteCosTheta(
  y0: Float64Array,
  f0: Float64Array,
  y1: Float64Array,
  f1: Float64Array,
  h: number,
  sigma: number,
): number {
  const s2 = sigma * sigma;
  const s3 = s2 * sigma;
  const th =
    (2 * s3 - 3 * s2 + 1) * y0[ITH]! +
    (s3 - 2 * s2 + sigma) * h * f0[ITH]! +
    (-2 * s3 + 3 * s2) * y1[ITH]! +
    (s3 - s2) * h * f1[ITH]!;
  return Math.cos(th);
}

/**
 * Locate a cosθ sign change in [σLo, σHi] by bisection on the Hermite
 * interpolant. Returns the crossing σ, or -1 if the endpoint signs agree.
 */
function bisectCrossing(
  y0: Float64Array,
  f0: Float64Array,
  y1: Float64Array,
  f1: Float64Array,
  h: number,
  sigmaLo: number,
  sigmaHi: number,
): number {
  let lo = sigmaLo;
  let hi = sigmaHi;
  let cLo = hermiteCosTheta(y0, f0, y1, f1, h, lo);
  const cHi = hermiteCosTheta(y0, f0, y1, f1, h, hi);
  if (cLo === 0) return lo;
  if (cHi === 0) return hi;
  if (Math.sign(cLo) === Math.sign(cHi)) return -1;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    const cMid = hermiteCosTheta(y0, f0, y1, f1, h, mid);
    if (Math.sign(cMid) === Math.sign(cLo)) {
      lo = mid;
      cLo = cMid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

/** Integrate a null geodesic from initial state y0 (PHYSICS.md §4 layout) until termination. */
export function integrateGeodesic(
  a: number,
  y0: Float64Array,
  options: GeodesicOptions = {},
): GeodesicResult {
  const rtol = options.rtol ?? DEFAULTS.rtol;
  const atol = options.atol ?? DEFAULTS.atol;
  const maxSteps = options.maxSteps ?? DEFAULTS.maxSteps;
  const rEscape = options.rEscape ?? DEFAULTS.rEscape;
  const horizonEps = options.horizonEpsilon ?? DEFAULTS.horizonEpsilon;
  const monitorTol = options.monitorTolerance ?? DEFAULTS.monitorTolerance;
  const disk = options.disk ?? null;

  const rPlus = horizonRadius(a);
  const rCapture = rPlus * (1 + horizonEps);

  const scratch = createRhsScratch();
  const metricScratch = createMetric();

  const y = Float64Array.from(y0);
  const yTrial = new Float64Array(STATE_SIZE);
  const yStage = new Float64Array(STATE_SIZE);
  const yHit = new Float64Array(STATE_SIZE);
  const k1 = new Float64Array(STATE_SIZE);
  const k2 = new Float64Array(STATE_SIZE);
  const k3 = new Float64Array(STATE_SIZE);
  const k4 = new Float64Array(STATE_SIZE);
  const k5 = new Float64Array(STATE_SIZE);
  const k6 = new Float64Array(STATE_SIZE);
  const k7 = new Float64Array(STATE_SIZE);

  const q0 = carterConstant(a, y);
  const qScale = Math.max(Math.abs(q0), 1);

  let lambda = 0;
  let steps = 0;
  let rejectedSteps = 0;
  let maxAbsH = hamiltonianResidual(a, y, scratch);
  let maxDriftQ = 0;
  let errOld = 1e-4;
  let monitorRejectArmed = true;

  geodesicRhs(a, y, k1, scratch);

  // Initial step from RHS scale (adaptivity corrects any misestimate).
  let h = Infinity;
  for (let i = 0; i < STATE_SIZE; i++) {
    const sc = atol + rtol * Math.abs(y[i]!);
    const fi = Math.abs(k1[i]!);
    if (fi > 0) h = Math.min(h, sc / fi);
  }
  h = Math.min(Math.max(h * 1e3, 1e-8), 1.0);

  const result = (status: RayStatus, state: Float64Array): GeodesicResult => ({
    status,
    state: Float64Array.from(state),
    lambda,
    steps,
    rejectedSteps,
    maxAbsH,
    maxDriftQ,
  });

  while (steps < maxSteps) {
    const r = y[IR]!;

    // --- termination checks on the current accepted state ---
    if (r <= rCapture) return result('CAPTURED', y);
    if (r >= rEscape && k1[IR]! > 0) {
      const m = evalMetric(a, r, y[ITH]!, metricScratch);
      const res = result('ESCAPED', y);
      res.escapeDirection = cartesianDirection(
        a, r, y[ITH]!, y[IPH]!,
        m.grr * y[5]!, m.gthth * y[6]!, m.gtph * y[4]! + m.gphph * y[7]!,
      );
      return res;
    }

    // --- step clamps (PHYSICS.md §10) ---
    let hMax = Infinity;
    const drAbs = Math.abs(k1[IR]!);
    if (k1[IR]! < 0 && drAbs > 0) {
      // Never let any stage evaluate below the horizon: bound |Δr| to half
      // the remaining gap to the capture radius.
      hMax = Math.min(hMax, Math.max(0.5 * (r - rCapture), 1e-10) / drAbs);
    }
    if (disk && r > 0.5 * disk.rInner && r < 2 * disk.rOuter) {
      const dthAbs = Math.abs(k1[ITH]!);
      if (dthAbs > 0) hMax = Math.min(hMax, THETA_STEP_CAP / dthAbs);
    }
    if (h > hMax) h = hMax;

    // --- Dormand-Prince stages (k1 is fresh or FSAL-reused) ---
    for (let i = 0; i < STATE_SIZE; i++) yStage[i] = y[i]! + h * A21 * k1[i]!;
    geodesicRhs(a, yStage, k2, scratch);
    for (let i = 0; i < STATE_SIZE; i++) yStage[i] = y[i]! + h * (A31 * k1[i]! + A32 * k2[i]!);
    geodesicRhs(a, yStage, k3, scratch);
    for (let i = 0; i < STATE_SIZE; i++) {
      yStage[i] = y[i]! + h * (A41 * k1[i]! + A42 * k2[i]! + A43 * k3[i]!);
    }
    geodesicRhs(a, yStage, k4, scratch);
    for (let i = 0; i < STATE_SIZE; i++) {
      yStage[i] = y[i]! + h * (A51 * k1[i]! + A52 * k2[i]! + A53 * k3[i]! + A54 * k4[i]!);
    }
    geodesicRhs(a, yStage, k5, scratch);
    for (let i = 0; i < STATE_SIZE; i++) {
      yStage[i] = y[i]! + h * (A61 * k1[i]! + A62 * k2[i]! + A63 * k3[i]! + A64 * k4[i]! + A65 * k5[i]!);
    }
    geodesicRhs(a, yStage, k6, scratch);
    for (let i = 0; i < STATE_SIZE; i++) {
      yTrial[i] = y[i]! + h * (B1 * k1[i]! + B3 * k3[i]! + B4 * k4[i]! + B5 * k5[i]! + B6 * k6[i]!);
    }
    geodesicRhs(a, yTrial, k7, scratch);

    // --- embedded error estimate ---
    let errSq = 0;
    for (let i = 0; i < STATE_SIZE; i++) {
      const e = h * (E1 * k1[i]! + E3 * k3[i]! + E4 * k4[i]! + E5 * k5[i]! + E6 * k6[i]! + E7 * k7[i]!);
      const sc = atol + rtol * Math.max(Math.abs(y[i]!), Math.abs(yTrial[i]!));
      errSq += (e / sc) * (e / sc);
    }
    const err = Math.sqrt(errSq / STATE_SIZE);

    if (err > 1) {
      rejectedSteps++;
      h *= Math.max(MIN_SCALE, SAFETY * Math.pow(err, -0.2));
      continue;
    }

    // --- monitors on the trial state (PHYSICS.md §10) ---
    const absH = hamiltonianResidual(a, yTrial, scratch);
    const driftQ = Math.abs(carterConstant(a, yTrial) - q0) / qScale;
    if ((absH > monitorTol || driftQ > monitorTol) && monitorRejectArmed && h > 1e-12) {
      // One retry at half step; if the breach persists it is recorded, not fatal.
      monitorRejectArmed = false;
      rejectedSteps++;
      h *= 0.5;
      continue;
    }
    monitorRejectArmed = true;
    if (absH > maxAbsH) maxAbsH = absH;
    if (driftQ > maxDriftQ) maxDriftQ = driftQ;

    // --- disk-plane crossing detection on the accepted step ---
    if (disk) {
      // Check both half-intervals so a double crossing within one step
      // (enter + exit the plane) cannot be missed.
      const brackets: Array<[number, number]> = [[0, 0.5], [0.5, 1]];
      for (const [lo, hi] of brackets) {
        const sigma = bisectCrossing(y, k1, yTrial, k7, h, lo, hi);
        if (sigma < 0) continue;
        hermiteState(y, k1, yTrial, k7, h, sigma, yHit);
        const rHit = yHit[IR]!;
        if (rHit >= disk.rInner && rHit <= disk.rOuter) {
          lambda += sigma * h;
          steps++;
          return result('HIT_DISK', yHit);
        }
      }
    }

    // --- accept ---
    y.set(yTrial);
    k1.set(k7); // FSAL
    lambda += h;
    steps++;

    const scale = Math.min(
      MAX_SCALE,
      Math.max(MIN_SCALE, SAFETY * Math.pow(err, -PI_ALPHA) * Math.pow(errOld, PI_BETA)),
    );
    h *= scale;
    errOld = Math.max(err, 1e-10);
  }

  return result('MAX_STEPS', y);
}
