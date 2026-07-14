/**
 * Ring 2 validation (VALIDATION.md §2): the float32 WGSL marcher confronted
 * with the float64 oracle on identical rays.
 *
 *  - V-O1: angular deviation of escape directions vs impact parameter,
 *    including near-critical rays (reported, not bounded, below b−b_c = 1e−2).
 *  - V-O4: conservation drift of the f32 integration, measured in f64 from
 *    the f32-quantized initial state to the returned final state.
 *  - Shadow diameter (M2 visual criterion): captured-pixel silhouette on the
 *    center scanline vs the Synge prediction at known FOV/distance.
 *
 * Dev-build only (dynamically imported through the __lensforge hook); never
 * part of the production bundle (ARCHITECTURE.md §2).
 */
import {
  carterConstant,
  createRhsScratch,
  hamiltonianResidual,
  STATE_SIZE,
} from '../physics/hamiltonian';
import { integrateGeodesic } from '../physics/integrator';
import { createMetric, evalMetric } from '../physics/metric';
import { equatorialPhoton, photonFromTetrad } from '../physics/photon';
import { orthonormalTetrad, zamoFourVelocity } from '../physics/tetrad';
import type { Renderer } from '../renderer/renderer';
import {
  STATUS_CAPTURED,
  STATUS_ESCAPED,
  type TracePass,
} from '../renderer/tracePass';
import type { OrbitCamera } from '../scene/orbitCamera';

const B_CRIT = 3 * Math.sqrt(3);
const R_START = 100;
/** V-O1 bound applies at b − b_c ≥ 1e−2 (VALIDATION.md §2). */
const BOUNDED_OFFSET = 1e-2;
/** High-tier V-O1 bound, arcminutes. */
const ARCMIN_BOUND_HIGH = 1;
/** V-O4 drift bound. */
const DRIFT_BOUND = 1e-4;

const ORACLE_STATUS_NAME: Record<number, string> = {
  [STATUS_CAPTURED]: 'CAPTURED',
  [STATUS_ESCAPED]: 'ESCAPED',
};

export interface VORow {
  label: string;
  bOffset: number | null;
  oracleStatus: string;
  gpuStatus: string;
  statusAgrees: boolean;
  arcmin: number | null;
  gpuSteps: number;
  driftQ: number;
  residualH: number;
}

export interface VOReport {
  rows: VORow[];
  maxArcminBounded: number;
  maxArcminNearCritical: number;
  maxDriftQ: number;
  /** H residual over ESCAPED rays (well-conditioned endpoint). */
  maxResidualH: number;
  /** H residual over CAPTURED rays — conditioning-limited near the horizon
   * (≈ ε_f32/Δ ~ 1e-4 at the capture radius); reported, not bounded. */
  maxResidualHCaptured: number;
  statusAgreementBounded: boolean;
  dispatches: number;
  pass: boolean;
}

interface TestRay {
  label: string;
  bOffset: number | null;
  state: Float64Array;
}

/**
 * Angle between two near-parallel directions via the chord, 2·asin(|d̂−ô|/2).
 * acos(dot) is unusable here: the GPU vector is normalized in f32, and its
 * ~1e-7 norm error alone inflates acos near dot = 1 to √(2e-7) ≈ 1.5 arcmin —
 * a metric artifact an order above the real deviation. Renormalizing in f64
 * and using the chord keeps full precision down to micro-arcminutes.
 */
function angularSeparationArcmin(d: ArrayLike<number>, o: ArrayLike<number>): number {
  const n = Math.hypot(d[0]!, d[1]!, d[2]!);
  const dx = d[0]! / n - o[0]!;
  const dy = d[1]! / n - o[1]!;
  const dz = d[2]! / n - o[2]!;
  const chord = Math.hypot(dx, dy, dz);
  return (2 * Math.asin(Math.min(1, chord / 2)) * 180 * 60) / Math.PI;
}

function buildRaySet(): TestRay[] {
  const rays: TestRay[] = [];
  for (const off of [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 2, 5, 10]) {
    rays.push({ label: `escape b_c+${off}`, bOffset: off, state: equatorialPhoton(0, R_START, B_CRIT + off, true) });
  }
  for (const off of [1e-3, 1e-2, 0.1, 1, 2]) {
    rays.push({ label: `capture b_c-${off}`, bOffset: -off, state: equatorialPhoton(0, R_START, B_CRIT - off, true) });
  }
  // Inclined tetrad rays (Q ≠ 0) — drift coverage off the equatorial plane.
  const m = createMetric();
  evalMetric(0, R_START, Math.PI / 3, m);
  const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, 0, R_START));
  for (const n of [
    [-1, 0.03, 0.02],
    [-1, -0.05, 0.04],
    [-1, 0.06, -0.01],
  ] as Array<[number, number, number]>) {
    rays.push({
      label: `inclined n=(${n.join(',')})`,
      bOffset: null,
      state: photonFromTetrad(m, tetrad, n, { r: R_START, theta: Math.PI / 3 }),
    });
  }
  return rays;
}

export async function runVOSuite(tracePass: TracePass): Promise<VOReport> {
  const rays = buildRaySet();
  const batch = new Float64Array(rays.length * STATE_SIZE);
  rays.forEach((ray, i) => batch.set(ray.state, i * STATE_SIZE));

  // Oracle integrates the same f32-quantized initial states the GPU receives,
  // so the comparison isolates integration error from input quantization.
  const quantized = Float64Array.from(batch, Math.fround);
  const gpu = await tracePass.runBatch(batch, 0);

  const scratch = createRhsScratch();
  const rows: VORow[] = [];
  let maxArcminBounded = 0;
  let maxArcminNearCritical = 0;
  let maxDriftQ = 0;
  let maxResidualH = 0;
  let maxResidualHCaptured = 0;
  let statusAgreementBounded = true;

  rays.forEach((ray, i) => {
    const y0 = quantized.subarray(i * STATE_SIZE, (i + 1) * STATE_SIZE);
    const oracle = integrateGeodesic(0, Float64Array.from(y0), { rtol: 1e-13, atol: 1e-15 });
    const gpuStatus = gpu.statuses[i]!;
    const gpuFinal = gpu.states.subarray(i * STATE_SIZE, (i + 1) * STATE_SIZE) as Float64Array;

    const oracleName = ORACLE_STATUS_NAME[oracle.status === 'CAPTURED' ? STATUS_CAPTURED : STATUS_ESCAPED] ?? oracle.status;
    const gpuName =
      gpuStatus === STATUS_CAPTURED ? 'CAPTURED' : gpuStatus === STATUS_ESCAPED ? 'ESCAPED' : `code ${gpuStatus}`;
    const statusAgrees = oracleName === gpuName;

    let arcmin: number | null = null;
    if (oracle.status === 'ESCAPED' && gpuStatus === STATUS_ESCAPED && oracle.escapeDirection) {
      arcmin = angularSeparationArcmin(
        gpu.directions.subarray(i * 3, i * 3 + 3),
        oracle.escapeDirection,
      );
    }

    const q0 = carterConstant(0, Float64Array.from(y0));
    const driftQ = Math.abs(carterConstant(0, gpuFinal) - q0) / Math.max(1, Math.abs(q0));
    const residualH = hamiltonianResidual(0, gpuFinal, scratch);

    const bounded = ray.bOffset === null || Math.abs(ray.bOffset) >= BOUNDED_OFFSET;
    if (bounded) {
      if (arcmin !== null) maxArcminBounded = Math.max(maxArcminBounded, arcmin);
      if (!statusAgrees) statusAgreementBounded = false;
      maxDriftQ = Math.max(maxDriftQ, driftQ);
      if (gpuStatus === STATUS_ESCAPED) maxResidualH = Math.max(maxResidualH, residualH);
      else maxResidualHCaptured = Math.max(maxResidualHCaptured, residualH);
    } else if (arcmin !== null) {
      maxArcminNearCritical = Math.max(maxArcminNearCritical, arcmin);
    }

    rows.push({
      label: ray.label,
      bOffset: ray.bOffset,
      oracleStatus: oracleName,
      gpuStatus: gpuName,
      statusAgrees,
      arcmin,
      gpuSteps: gpu.steps[i]!,
      driftQ,
      residualH,
    });
  });

  return {
    rows,
    maxArcminBounded,
    maxArcminNearCritical,
    maxDriftQ,
    maxResidualH,
    maxResidualHCaptured,
    statusAgreementBounded,
    dispatches: gpu.dispatches,
    pass:
      statusAgreementBounded &&
      maxArcminBounded <= ARCMIN_BOUND_HIGH &&
      maxDriftQ <= DRIFT_BOUND &&
      maxResidualH <= DRIFT_BOUND,
  };
}

export interface ShadowMeasurement {
  measuredPx: number;
  predictedPx: number;
  relError: number;
  framesToConverge: number;
}

/** Shadow-diameter criterion (IMPLEMENTATION.md M2): silhouette vs Synge at known FOV/distance. */
export async function measureShadowDiameter(
  renderer: Renderer,
  camera: OrbitCamera,
): Promise<ShadowMeasurement> {
  const r0 = 1000;
  const fovY = 0.03;
  camera.set({ r: r0, theta: Math.PI / 2, phi: 0, fovY });

  let frames = 0;
  for (; frames < 300; frames++) {
    renderer.renderFrame();
    await renderer.onGpuIdle();
    await new Promise((r) => setTimeout(r, 0)); // let the counter mapAsync settle
    if (renderer.tracePass.getKnownMarching() === 0) break;
  }

  const statuses = await renderer.tracePass.readScreenStatuses();
  const { width, height } = renderer.dimensions();
  const row = Math.floor(height / 2);
  let first = -1;
  let last = -1;
  for (let x = 0; x < width; x++) {
    if (statuses[row * width + x] === STATUS_CAPTURED) {
      if (first < 0) first = x;
      last = x;
    }
  }
  const measuredPx = first < 0 ? 0 : last - first + 1;

  const alphaShadow = Math.asin((B_CRIT * Math.sqrt(1 - 2 / r0)) / r0);
  const tanHalfFovY = Math.tan(fovY / 2);
  const predictedPx = (2 * Math.tan(alphaShadow) * (height / 2)) / tanHalfFovY;

  return {
    measuredPx,
    predictedPx,
    relError: Math.abs(measuredPx - predictedPx) / predictedPx,
    framesToConverge: frames,
  };
}
