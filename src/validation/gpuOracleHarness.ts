/**
 * Ring 2 + Ring 3 validation (VALIDATION.md §2, §3): the float32 WGSL marcher
 * confronted with the float64 oracle on identical rays, at any spin.
 *
 *  - V-O1: angular deviation of escape directions vs impact parameter
 *    (prograde and retrograde sweeps around the spin-dependent b_c±),
 *    including near-critical rays (reported, not bounded, below |b−b_c| = 1e−2).
 *  - V-O4: conservation drift of the f32 integration, measured in f64 from
 *    the f32-quantized initial state to the returned final state.
 *  - Shadow diameter (M2 visual criterion, a = 0): captured-pixel silhouette
 *    on the center scanline vs the Synge prediction at known FOV/distance.
 *  - V-I1 (M3): rendered shadow silhouette vs the Bardeen analytic contour
 *    mapped exactly onto the screen (src/validation/shadowContour.ts).
 *
 * Dev-build only (dynamically imported through the __lensforge hook); never
 * part of the production bundle (ARCHITECTURE.md §2).
 */
import {
  carterConstant,
  createRhsScratch,
  hamiltonianResidual,
  IPH,
  IPPH,
  IPT,
  IR,
  STATE_SIZE,
} from '../physics/hamiltonian';
import { integrateGeodesic } from '../physics/integrator';
import { createMetric, evalMetric } from '../physics/metric';
import { angularVelocity, circularFourVelocity, circularUt, iscoRadius } from '../physics/orbits';
import { equatorialPhoton, photonFromTetrad, polarPhoton } from '../physics/photon';
import { criticalImpactParameter } from '../physics/shadow';
import { orthonormalTetrad, zamoFourVelocity, type Tetrad } from '../physics/tetrad';
import { sampleNtSlice, samplePlanckXyz } from '../renderer/diskLuts';
import type { Renderer } from '../renderer/renderer';
import {
  STATUS_CAPTURED,
  STATUS_ESCAPED,
  STATUS_HIT_DISK,
  STATUS_MAX_STEPS,
  type TracePass,
} from '../renderer/tracePass';
import type { OrbitCamera } from '../scene/orbitCamera';
import { VI1_FRAMING, VI3_FRAMING } from '../scene/presets';
import { analyticShadowContourPixels, type CameraFraming } from './shadowContour';
import { computeSsim, rgba8ToLuma } from './ssim';

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

function buildRaySet(spin: number): TestRay[] {
  const rays: TestRay[] = [];
  const bcPro = criticalImpactParameter(spin, true);
  for (const off of [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 2, 5, 10]) {
    rays.push({ label: `escape pro b_c+${off}`, bOffset: off, state: equatorialPhoton(spin, R_START, bcPro + off, true) });
  }
  for (const off of [1e-3, 1e-2, 0.1, 1, 2]) {
    rays.push({ label: `capture pro b_c-${off}`, bOffset: -off, state: equatorialPhoton(spin, R_START, bcPro - off, true) });
  }
  if (spin > 0) {
    // Retrograde sweep: capture side is |b| < |b_c| with b_c < 0.
    const bcRetro = criticalImpactParameter(spin, false);
    for (const off of [1e-3, 1e-2, 0.1, 1, 5]) {
      rays.push({ label: `escape retro b_c-${off}`, bOffset: off, state: equatorialPhoton(spin, R_START, bcRetro - off, true) });
    }
    for (const off of [1e-3, 1e-2, 0.1, 1]) {
      rays.push({ label: `capture retro b_c+${off}`, bOffset: -off, state: equatorialPhoton(spin, R_START, bcRetro + off, true) });
    }
  }
  // Inclined tetrad rays (Q ≠ 0) — drift coverage off the equatorial plane.
  const m = createMetric();
  evalMetric(spin, R_START, Math.PI / 3, m);
  const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, spin, R_START));
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
  // Polar-axis rays (p_φ = 0): exercise the axis-regular φφ path in WGSL —
  // an escaping ray crossing the pole and a captured near-axis plunge.
  rays.push({ label: 'polar cross √Q=8', bOffset: null, state: polarPhoton(spin, R_START, Math.PI / 2, -8, true) });
  rays.push({ label: 'polar plunge √Q=4', bOffset: null, state: polarPhoton(spin, R_START, Math.PI / 2, -4, true) });
  return rays;
}

export async function runVOSuite(tracePass: TracePass, spin = 0): Promise<VOReport> {
  const rays = buildRaySet(spin);
  const batch = new Float64Array(rays.length * STATE_SIZE);
  rays.forEach((ray, i) => batch.set(ray.state, i * STATE_SIZE));

  // Oracle integrates the same f32-quantized initial states the GPU receives,
  // so the comparison isolates integration error from input quantization.
  const quantized = Float64Array.from(batch, Math.fround);
  const gpu = await tracePass.runBatch(batch, spin);

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
    const oracle = integrateGeodesic(spin, Float64Array.from(y0), { rtol: 1e-13, atol: 1e-15 });
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

    const q0 = carterConstant(spin, Float64Array.from(y0));
    const driftQ = Math.abs(carterConstant(spin, gpuFinal) - q0) / Math.max(1, Math.abs(q0));
    const residualH = hamiltonianResidual(spin, gpuFinal, scratch);

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

/** Shadow-diameter criterion (IMPLEMENTATION.md M2): silhouette vs Synge at known FOV/distance.
 * Runs with the disk disabled — disk hits would occlude the silhouette. */
export async function measureShadowDiameter(
  renderer: Renderer,
  camera: OrbitCamera,
): Promise<ShadowMeasurement> {
  const savedDisk = renderer.tracePass.getDisk();
  renderer.tracePass.setDisk({ ...savedDisk, enabled: false });
  // Refinement rounds re-init the persistent ray state with jittered rays;
  // measurements are defined on the base (pixel-center) sample.
  renderer.tracePass.setRefinementEnabled(false);
  try {
    return await measureShadowDiameterInner(renderer, camera);
  } finally {
    renderer.tracePass.setRefinementEnabled(true);
    renderer.tracePass.setDisk(savedDisk);
  }
}

async function measureShadowDiameterInner(
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

// ---------------------------------------------------------------------------
// V-O2 (VALIDATION.md §2): disk-hit deviation — GPU (r, φ, g) vs float64
// oracle on identical f32-quantized rays. Bounds (High tier): Δr < 1e-3,
// Δg/g < 1e-3; Δφ reported (compared mod 2π).
// ---------------------------------------------------------------------------

const VO2_R_OUT = 20;
/** Bound on the incidence-normalized Δr. The f32 transverse ray error is the
 * V-O1 budget (≲ 1 arcmin ⇒ ~3e-2 over the ~120M path of the reference
 * framing); measured worst 2.9e-3. 1e-2 sits between measurement and the
 * V-O1-derived ceiling — the raw Δr additionally carries the published
 * incidence amplification (VALIDATION.md §2). */
const VO2_DR_BOUND = 1e-2;
const VO2_DG_BOUND = 1e-3;

export interface VO2Row {
  label: string;
  oracleStatus: string;
  gpuStatus: string;
  statusAgrees: boolean;
  rOracle: number | null;
  dr: number | null;
  /** Δr with the grazing-incidence amplification divided out: the crossing
   * maps transverse ray error into r by c = |Δ·p_r| / (r·|p_θ|) (the radial
   * slope of the ray through the plane), so dr/max(1, c) is the
   * transverse-scale deviation the V-O1-style bound actually constrains. */
  drNormalized: number | null;
  incidenceAmplification: number | null;
  /** Distance of the oracle hit to the nearer disk edge (ISCO or r_out):
   * rays inside EDGE_MARGIN can legitimately differ in status between f32
   * and f64 (the hit/miss boundary itself is the sensitive quantity). */
  edgeMargin: number | null;
  dphiWrapped: number | null;
  dgRel: number | null;
  /** φ winding of the oracle hit — |φ| > π marks a wrapped (higher-order) image ray. */
  phiOracle: number | null;
}

export interface VO2Report {
  spin: number;
  rows: VO2Row[];
  hits: number;
  wrappedHits: number;
  /** Rays whose oracle hit lies within EDGE_MARGIN of a disk edge — reported,
   * not bounded (cf. V-O1 near-critical waiver). */
  edgeRays: number;
  maxDr: number;
  maxDrNormalized: number;
  maxDphiWrapped: number;
  maxDgRel: number;
  statusAgreementBounded: boolean;
  pass: boolean;
}

/** Oracle hits closer than this to the ISCO/outer edge are status-fragile. */
const VO2_EDGE_MARGIN = 1e-2;

function wrapAngle(x: number): number {
  const w = x % (2 * Math.PI);
  return w > Math.PI ? w - 2 * Math.PI : w < -Math.PI ? w + 2 * Math.PI : w;
}

/**
 * The f32 ray-init contract of Pass A (init.wgsl, AXIAL_SNAP_SMIN,
 * PHYSICS.md §10): rays whose polar turning layer s_min = |p_φ|/P is below
 * f32 θ-resolution are quantized onto the exact L = 0 axis branch. Batch rays
 * bypass Pass A, so the harness applies the same quantization before the
 * GPU/oracle comparison — otherwise it would feed rays the f32 pipeline is
 * documented not to accept (e.g. p_φ ~ 1e-18 from f64 roundoff of the ZAMO
 * u_φ = 0, whose s_min ~ 1e-19 whip even float64 needs ~10⁴ steps to grind
 * through).
 */
const AXIAL_SNAP_SMIN = 4e-5;
function applyAxialSnap(state: Float64Array): Float64Array {
  const s = Math.sin(state[2]!);
  const lOverS = state[IPPH]! / s;
  const P = Math.hypot(state[6]!, lOverS);
  if (Math.abs(state[IPPH]!) < AXIAL_SNAP_SMIN * P) state[IPPH] = 0;
  return state;
}

/** Exact redshift for a tetrad-launched photon hitting the disk at r (PHYSICS.md §8). */
function diskG(a: number, r: number, E: number, L: number): number {
  return 1 / (circularUt(a, r) * (E - angularVelocity(a, r) * L));
}

export async function runVO2Suite(tracePass: TracePass, spin = 0): Promise<VO2Report> {
  const rIsco = iscoRadius(spin);
  const disk = { rInner: rIsco, rOuter: VO2_R_OUT };
  const r0 = 100;
  const th0 = 1.309; // ~75° inclination: primary hits and near-shadow wrapped rays
  const m = createMetric();
  evalMetric(spin, r0, th0, m);
  const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, spin, r0));

  // Direction scan: the oracle decides which rays hit; the hit set (spread in
  // r_hit, plus every wrapped ray) becomes the GPU comparison batch.
  const candidates: Array<{ label: string; state: Float64Array; oracle: ReturnType<typeof integrateGeodesic> }> = [];
  const angles = [-0.2, -0.12, -0.08, -0.055, -0.04, -0.03, 0.03, 0.04, 0.055, 0.08, 0.12, 0.2];
  for (const n2 of angles) {
    for (const n3 of [-0.08, -0.045, -0.03, 0, 0.03, 0.045, 0.08]) {
      const state = photonFromTetrad(m, tetrad, [-1, n2, n3], { r: r0, theta: th0 });
      const quantized = applyAxialSnap(Float64Array.from(state, Math.fround));
      const oracle = integrateGeodesic(spin, Float64Array.from(quantized), {
        rtol: 1e-13,
        atol: 1e-15,
        disk,
      });
      if (oracle.status === 'HIT_DISK') {
        candidates.push({ label: `n=(${n2},${n3})`, state: quantized, oracle });
      }
    }
  }
  // Keep a manageable, representative set: sort by hit radius, stride-sample,
  // and always keep wrapped-image rays.
  candidates.sort((p, q) => p.oracle.state[IR]! - q.oracle.state[IR]!);
  const wrapped = candidates.filter((c) => Math.abs(c.oracle.state[IPH]!) > Math.PI);
  const stride = Math.max(1, Math.floor(candidates.length / 14));
  const chosen = candidates.filter((_, i) => i % stride === 0);
  for (const w of wrapped) if (!chosen.includes(w)) chosen.push(w);

  const batch = new Float64Array(chosen.length * STATE_SIZE);
  chosen.forEach((c, i) => batch.set(c.state, i * STATE_SIZE));
  const gpu = await tracePass.runBatch(batch, spin, 400, { rIsco, rOuter: VO2_R_OUT });

  const rows: VO2Row[] = [];
  let maxDr = 0;
  let maxDrNormalized = 0;
  let maxDphiWrapped = 0;
  let maxDgRel = 0;
  let statusAgreementBounded = true;
  let hits = 0;
  let wrappedHits = 0;
  let edgeRays = 0;

  chosen.forEach((c, i) => {
    const gpuStatus = gpu.statuses[i]!;
    const statusAgrees = gpuStatus === STATUS_HIT_DISK;
    const rOracle = c.oracle.state[IR]!;
    const phiOracle = c.oracle.state[IPH]!;
    const edgeMargin = Math.min(rOracle - rIsco, VO2_R_OUT - rOracle);
    const isEdgeRay = edgeMargin < VO2_EDGE_MARGIN;
    if (isEdgeRay) edgeRays++;
    // Grazing-incidence conditioning of the crossing (see VO2Row docs).
    const pr = c.oracle.state[5]!;
    const pth = c.oracle.state[6]!;
    const Delta = rOracle * rOracle - 2 * rOracle + spin * spin;
    const incidenceAmplification = Math.abs(Delta * pr) / Math.max(rOracle * Math.abs(pth), 1e-30);

    if (!statusAgrees && !isEdgeRay) statusAgreementBounded = false;
    let dr: number | null = null;
    let drNormalized: number | null = null;
    let dphiWrapped: number | null = null;
    let dgRel: number | null = null;
    if (statusAgrees) {
      hits++;
      if (Math.abs(phiOracle) > Math.PI) wrappedHits++;
      const rGpu = gpu.directions[i * 3]!;
      const phiGpu = gpu.directions[i * 3 + 1]!;
      const gGpu = gpu.directions[i * 3 + 2]!;
      const E = -c.state[IPT]!;
      const L = c.state[IPPH]!;
      const gOracle = diskG(spin, rOracle, E, L);
      dr = Math.abs(rGpu - rOracle);
      drNormalized = dr / Math.max(1, incidenceAmplification);
      dphiWrapped = Math.abs(wrapAngle(phiGpu - phiOracle));
      dgRel = Math.abs(gGpu - gOracle) / gOracle;
      maxDr = Math.max(maxDr, dr);
      maxDrNormalized = Math.max(maxDrNormalized, drNormalized);
      maxDphiWrapped = Math.max(maxDphiWrapped, dphiWrapped);
      maxDgRel = Math.max(maxDgRel, dgRel);
    }
    rows.push({
      label: c.label,
      oracleStatus: 'HIT_DISK',
      gpuStatus:
        gpuStatus === STATUS_HIT_DISK
          ? 'HIT_DISK'
          : (ORACLE_STATUS_NAME[gpuStatus] ?? `code ${gpuStatus}`),
      statusAgrees,
      rOracle,
      dr,
      drNormalized,
      incidenceAmplification,
      edgeMargin,
      dphiWrapped,
      dgRel,
      phiOracle,
    });
  });

  return {
    spin,
    rows,
    hits,
    wrappedHits,
    edgeRays,
    maxDr,
    maxDrNormalized,
    maxDphiWrapped,
    maxDgRel,
    statusAgreementBounded,
    pass:
      statusAgreementBounded &&
      hits >= 8 &&
      maxDrNormalized <= VO2_DR_BOUND &&
      maxDgRel <= VO2_DG_BOUND,
  };
}

export interface ShadowContourReport {
  spin: number;
  width: number;
  height: number;
  boundaryPoints: number;
  /** Max/mean distance of measured silhouette boundary to the analytic contour, px. */
  maxDeviationPx: number;
  meanDeviationPx: number;
  /** Equatorial-row silhouette extent [left, right] in px (D-shape displacement check). */
  equatorRowExtent: [number, number] | null;
  maxStepsPixels: number;
  framesToConverge: number;
  pass: boolean;
}

/**
 * V-I1 (VALIDATION.md §3): rendered Kerr shadow silhouette vs Bardeen's
 * analytic contour [R1], mapped exactly onto the screen through the same
 * float64 ZAMO tetrad the renderer uses (no far-field approximation; the
 * critical curve in (ξ, η) space is observer-independent). The caller must
 * have routed `spin` through the camera path before invoking (dev hook does).
 */
export async function measureShadowContour(
  renderer: Renderer,
  camera: OrbitCamera,
  spin: number,
): Promise<ShadowContourReport> {
  const savedDisk = renderer.tracePass.getDisk();
  renderer.tracePass.setDisk({ ...savedDisk, enabled: false });
  renderer.tracePass.setRefinementEnabled(false); // base-sample statuses only
  try {
    return await measureShadowContourInner(renderer, camera, spin);
  } finally {
    renderer.tracePass.setRefinementEnabled(true);
    renderer.tracePass.setDisk(savedDisk);
  }
}

async function measureShadowContourInner(
  renderer: Renderer,
  camera: OrbitCamera,
  spin: number,
): Promise<ShadowContourReport> {
  camera.set({ ...VI1_FRAMING });

  let frames = 0;
  for (; frames < 400; frames++) {
    renderer.renderFrame();
    await renderer.onGpuIdle();
    await new Promise((r) => setTimeout(r, 0)); // let the counter mapAsync settle
    if (renderer.tracePass.getKnownMarching() === 0) break;
  }

  const statuses = await renderer.tracePass.readScreenStatuses();
  const { width, height } = renderer.dimensions();
  const framing: CameraFraming = { r: VI1_FRAMING.r, theta: VI1_FRAMING.theta, fovY: VI1_FRAMING.fovY, width, height };

  // Silhouette boundary: subpixel midpoints of strict CAPTURED↔ESCAPED
  // transitions along rows and columns (MAX_STEPS neighbours are excluded —
  // they carry no capture information and are counted separately).
  const boundary: Array<{ x: number; y: number }> = [];
  let maxStepsPixels = 0;
  for (let i = 0; i < width * height; i++) {
    if (statuses[i] === STATUS_MAX_STEPS) maxStepsPixels++;
  }
  const at = (x: number, y: number): number => statuses[y * width + x]!;
  for (let y = 0; y < height; y++) {
    for (let x = 1; x < width; x++) {
      const s0 = at(x - 1, y);
      const s1 = at(x, y);
      if (s0 === STATUS_ESCAPED && s1 === STATUS_CAPTURED) boundary.push({ x: x - 0.5, y });
      else if (s0 === STATUS_CAPTURED && s1 === STATUS_ESCAPED) boundary.push({ x: x - 0.5, y });
    }
  }
  for (let x = 0; x < width; x++) {
    for (let y = 1; y < height; y++) {
      const s0 = at(x, y - 1);
      const s1 = at(x, y);
      if (
        (s0 === STATUS_ESCAPED && s1 === STATUS_CAPTURED) ||
        (s0 === STATUS_CAPTURED && s1 === STATUS_ESCAPED)
      ) {
        boundary.push({ x, y: y - 0.5 });
      }
    }
  }

  const analytic = analyticShadowContourPixels(spin, framing, 4096);
  let maxDev = 0;
  let sumDev = 0;
  for (const b of boundary) {
    let best = Infinity;
    for (const p of analytic) {
      const dx = p.x - b.x;
      const dy = p.y - b.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < best) best = d2;
    }
    const d = Math.sqrt(best);
    if (d > maxDev) maxDev = d;
    sumDev += d;
  }

  const row = Math.floor(height / 2);
  let first = -1;
  let last = -1;
  for (let x = 0; x < width; x++) {
    if (at(x, row) === STATUS_CAPTURED) {
      if (first < 0) first = x;
      last = x;
    }
  }

  return {
    spin,
    width,
    height,
    boundaryPoints: boundary.length,
    maxDeviationPx: maxDev,
    meanDeviationPx: boundary.length > 0 ? sumDev / boundary.length : Infinity,
    equatorRowExtent: first >= 0 ? [first, last] : null,
    maxStepsPixels,
    framesToConverge: frames,
    pass: boundary.length > 0 && maxDev <= 1.0,
  };
}

// ---------------------------------------------------------------------------
// V-I3, oracle half (VALIDATION.md §3): beaming asymmetry of the rendered
// disk vs a float64 oracle shade of the same pixels. The full DNGR
// feature-position overlay is a release-gate step (M8); the 5 % asymmetry
// criterion against the oracle render is executable now and validates the
// entire GPU shading path (intersection → g → LUT sampling → XYZ → sRGB).
// ---------------------------------------------------------------------------

/** V-I3 reference framing: near-edge-on ZAMO at r = 60M (beaming-dominated). */
export { VI1_FRAMING, VI3_FRAMING };

export interface BeamingReport {
  spin: number;
  width: number;
  height: number;
  diskPixels: number;
  sampledLeft: number;
  sampledRight: number;
  /** Σ luminance approaching side / Σ luminance receding side, sampled pixels. */
  ratioGpu: number;
  ratioOracle: number;
  relDiff: number;
  /** Mean per-pixel |Y_gpu − Y_oracle| / Y_oracle over the sampled set. */
  meanPixelRelDiff: number;
  maxPixelRelDiff: number;
  statusMismatches: number;
  framesToConverge: number;
  pass: boolean;
}

export async function measureBeamingAsymmetry(
  renderer: Renderer,
  camera: OrbitCamera,
  spin: number,
): Promise<BeamingReport> {
  const tp = renderer.tracePass;
  const luts = tp.getLuts();
  if (!luts) throw new Error('disk LUTs not loaded');
  const savedDisk = tp.getDisk();
  if (!savedDisk.enabled) tp.setDisk({ ...savedDisk, enabled: true });
  tp.setRefinementEnabled(false); // base-sample statuses/HDR only
  try {
  camera.set({ ...VI3_FRAMING });

  let frames = 0;
  for (; frames < 400; frames++) {
    renderer.renderFrame();
    await renderer.onGpuIdle();
    await new Promise((r) => setTimeout(r, 0));
    if (tp.getKnownMarching() === 0) break;
  }

  const statuses = await tp.readScreenStatuses();
  const { data, width, height } = await renderer.readHdr();
  const shading = tp.getDiskShading();
  const slice = tp.getNtSlice();
  const pose = tp.getCamera();
  if (!shading || !slice || !pose) throw new Error('disk shading state unavailable');

  const m = createMetric();
  evalMetric(spin, pose.r, pose.theta, m);
  const tetrad: Tetrad = orthonormalTetrad(m, zamoFourVelocity(m, spin, pose.r));
  const tanY = Math.tan(pose.fovY / 2);
  const tanX = (tanY * width) / height;
  const disk = { rInner: shading.rIsco, rOuter: shading.rOuter };

  let diskPixels = 0;
  for (let i = 0; i < width * height; i++) if (statuses[i] === STATUS_HIT_DISK) diskPixels++;

  // Deterministic subsample grid over disk pixels; per-pixel oracle shade.
  let sumGpuL = 0, sumGpuR = 0, sumOrL = 0, sumOrR = 0;
  let sampledLeft = 0, sampledRight = 0, statusMismatches = 0;
  let sumPixelRelDiff = 0, maxPixelRelDiff = 0, compared = 0;
  for (let y = 0; y < height; y += 6) {
    for (let x = 0; x < width; x += 10) {
      if (statuses[y * width + x] !== STATUS_HIT_DISK) continue;
      const ndcX = (2 * (x + 0.5)) / width - 1;
      const ndcY = 1 - (2 * (y + 0.5)) / height;
      const n: [number, number, number] = [-1, -ndcY * tanY, ndcX * tanX];
      const len = Math.hypot(...n);
      const y0 = photonFromTetrad(
        m, tetrad,
        [n[0] / len, n[1] / len, n[2] / len],
        { r: pose.r, theta: pose.theta, phi: pose.phi },
      );
      const oracle = integrateGeodesic(spin, y0, { rtol: 1e-12, atol: 1e-14, disk });
      if (oracle.status !== 'HIT_DISK') {
        statusMismatches++;
        continue;
      }
      const rHit = oracle.state[IR]!;
      const g = diskG(spin, rHit, -y0[IPT]!, y0[IPPH]!);
      const tHat = sampleNtSlice(slice, luts.nt.uMax, shading.rIsco, rHit);
      const yOracle = samplePlanckXyz(luts.planck, g * tHat * shading.tempScaleK)[1] * shading.radianceScale;
      const px = (y * width + x) * 4;
      // Linear sRGB → CIE Y (middle row of the sRGB→XYZ matrix).
      const yGpu = 0.2126 * data[px]! + 0.7152 * data[px + 1]! + 0.0722 * data[px + 2]!;
      if (yOracle > 0) {
        const rel = Math.abs(yGpu - yOracle) / yOracle;
        sumPixelRelDiff += rel;
        maxPixelRelDiff = Math.max(maxPixelRelDiff, rel);
        compared++;
      }
      if (x < width / 2) {
        sumGpuL += yGpu; sumOrL += yOracle; sampledLeft++;
      } else {
        sumGpuR += yGpu; sumOrR += yOracle; sampledRight++;
      }
    }
  }

  if (!savedDisk.enabled) tp.setDisk(savedDisk);

  const ratioGpu = sumGpuL / sumGpuR;
  const ratioOracle = sumOrL / sumOrR;
  const relDiff = Math.abs(ratioGpu / ratioOracle - 1);
  return {
    spin,
    width,
    height,
    diskPixels,
    sampledLeft,
    sampledRight,
    ratioGpu,
    ratioOracle,
    relDiff,
    meanPixelRelDiff: compared > 0 ? sumPixelRelDiff / compared : Infinity,
    maxPixelRelDiff,
    statusMismatches,
    framesToConverge: frames,
    pass: sampledLeft > 20 && sampledRight > 20 && relDiff < 0.05,
  };
  } finally {
    tp.setRefinementEnabled(true);
  }
}

// ---------------------------------------------------------------------------
// M7 star-splat magnification spot check (IMPLEMENTATION.md M7): the splat
// magnification is |det J|⁻¹ with J finite-differenced from neighboring rays'
// escape directions (shaders/sky/sky.wgsl). This harness recomputes J from
// the GPU's returned directions (the exact arithmetic the shader uses) and
// confronts it with a float64 ray-bundle Jacobian: four neighboring pixel
// rays integrated by the oracle, centrally differenced. Agreement bounds the
// end-to-end magnification error of the analytic star pipeline.
// ---------------------------------------------------------------------------

export interface StarMagRow {
  x: number;
  y: number;
  /** Point magnification ω_lin² / |det J| from the oracle bundle. */
  muOracle: number;
  muGpu: number;
  relErr: number;
}

export interface StarMagReport {
  spin: number;
  width: number;
  height: number;
  samples: number;
  maxRelErr: number;
  meanRelErr: number;
  rows: StarMagRow[];
  framesToConverge: number;
  pass: boolean;
}

/** Tangent-basis 2×2 determinant of [jx | jy] at direction d (shader mirror). */
function tangentDet(d: [number, number, number], jx: number[], jy: number[]): number {
  const up = Math.abs(d[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const t1u = [
    up[1]! * d[2] - up[2]! * d[1],
    up[2]! * d[0] - up[0]! * d[2],
    up[0]! * d[1] - up[1]! * d[0],
  ];
  const n1 = Math.hypot(t1u[0]!, t1u[1]!, t1u[2]!);
  const t1 = [t1u[0]! / n1, t1u[1]! / n1, t1u[2]! / n1];
  const t2 = [
    d[1] * t1[2]! - d[2] * t1[1]!,
    d[2] * t1[0]! - d[0] * t1[2]!,
    d[0] * t1[1]! - d[1] * t1[0]!,
  ];
  const a = jx[0]! * t1[0]! + jx[1]! * t1[1]! + jx[2]! * t1[2]!;
  const c = jx[0]! * t2[0]! + jx[1]! * t2[1]! + jx[2]! * t2[2]!;
  const b = jy[0]! * t1[0]! + jy[1]! * t1[1]! + jy[2]! * t1[2]!;
  const dd = jy[0]! * t2[0]! + jy[1]! * t2[1]! + jy[2]! * t2[2]!;
  return a * dd - b * c;
}

export async function measureStarMagnification(
  renderer: Renderer,
  camera: OrbitCamera,
  spin: number,
): Promise<StarMagReport> {
  const tp = renderer.tracePass;
  const savedDisk = tp.getDisk();
  tp.setDisk({ ...savedDisk, enabled: false }); // sky-only frame
  tp.setRefinementEnabled(false); // base pixel-center sample
  try {
    camera.set({ ...VI1_FRAMING });
    let frames = 0;
    for (; frames < 400; frames++) {
      renderer.renderFrame();
      await renderer.onGpuIdle();
      await new Promise((r) => setTimeout(r, 0));
      if (tp.getKnownMarching() === 0) break;
    }

    const statuses = await tp.readScreenStatuses();
    const results = await tp.readScreenResults();
    const { width, height } = renderer.dimensions();
    const rayCount = width * height;
    const pose = tp.getCamera();
    if (!pose) throw new Error('no camera');

    const m = createMetric();
    evalMetric(spin, pose.r, pose.theta, m);
    const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, spin, pose.r));
    const tanY = Math.tan(pose.fovY / 2);
    const tanX = (tanY * width) / height;
    const omegaLin = (2 * tanY) / height; // small-angle px pitch at frame center

    const gpuDir = (i: number): [number, number, number] => [
      results[0 * rayCount + i]!,
      results[1 * rayCount + i]!,
      results[2 * rayCount + i]!,
    ];
    const escaped = (x: number, y: number): boolean =>
      x >= 0 && x < width && y >= 0 && y < height && statuses[y * width + x] === STATUS_ESCAPED;
    const oracleDir = (x: number, y: number): [number, number, number] => {
      const ndcX = (2 * (x + 0.5)) / width - 1;
      const ndcY = 1 - (2 * (y + 0.5)) / height;
      const n: [number, number, number] = [-1, -ndcY * tanY, ndcX * tanX];
      const len = Math.hypot(...n);
      const y0 = photonFromTetrad(
        m,
        tetrad,
        [n[0] / len, n[1] / len, n[2] / len],
        { r: pose.r, theta: pose.theta, phi: pose.phi },
      );
      const o = integrateGeodesic(spin, y0, { rtol: 1e-12, atol: 1e-14 });
      if (o.status !== 'ESCAPED' || !o.escapeDirection) throw new Error(`oracle ray (${x},${y}) not escaped`);
      return o.escapeDirection as [number, number, number];
    };

    // Sample a deterministic grid of pixels whose full 4-neighborhood escaped,
    // spanning far field and the strongly lensed annulus near the shadow.
    const rows: StarMagRow[] = [];
    let maxRelErr = 0;
    let sumRelErr = 0;
    for (let y = Math.floor(height * 0.15); y < height * 0.85 && rows.length < 24; y += Math.floor(height / 9)) {
      for (let x = Math.floor(width * 0.1); x < width * 0.9 && rows.length < 24; x += Math.floor(width / 7)) {
        if (
          !escaped(x, y) || !escaped(x - 1, y) || !escaped(x + 1, y) ||
          !escaped(x, y - 1) || !escaped(x, y + 1)
        ) continue;
        const i = y * width + x;
        const d = gpuDir(i);
        const jxG = gpuDir(i + 1).map((v, k) => 0.5 * (v - gpuDir(i - 1)[k]!));
        const jyG = gpuDir(i + width).map((v, k) => 0.5 * (v - gpuDir(i - width)[k]!));
        const detGpu = tangentDet(d, jxG, jyG);

        const oC = oracleDir(x, y);
        const oXp = oracleDir(x + 1, y);
        const oXm = oracleDir(x - 1, y);
        const oYp = oracleDir(x, y + 1);
        const oYm = oracleDir(x, y - 1);
        const jxO = oXp.map((v, k) => 0.5 * (v - oXm[k]!));
        const jyO = oYp.map((v, k) => 0.5 * (v - oYm[k]!));
        const detOracle = tangentDet(oC, jxO, jyO);
        if (Math.abs(detOracle) < 1e-30) continue;

        const muOracle = (omegaLin * omegaLin) / Math.abs(detOracle);
        const muGpu = (omegaLin * omegaLin) / Math.abs(detGpu);
        const relErr = Math.abs(muGpu / muOracle - 1);
        rows.push({ x, y, muOracle, muGpu, relErr });
        maxRelErr = Math.max(maxRelErr, relErr);
        sumRelErr += relErr;
      }
    }

    return {
      spin,
      width,
      height,
      samples: rows.length,
      maxRelErr,
      meanRelErr: rows.length > 0 ? sumRelErr / rows.length : Infinity,
      rows,
      framesToConverge: frames,
      pass: rows.length >= 10 && maxRelErr < 0.05,
    };
  } finally {
    tp.setRefinementEnabled(true);
    tp.setDisk(savedDisk);
  }
}

// ---------------------------------------------------------------------------
// V-I5 golden capture (M7 re-baseline): converged base-sample HDR statistics
// per shipped preset — sphere of comparison is pre-tonemap physical luminance
// (VALIDATION.md §3): full-frame mean + a 16×9 cell-mean grid. Baselines are
// stored in tests/golden/; comparisons are tolerance-based (D12).
// ---------------------------------------------------------------------------

export interface GoldenEntry {
  name: string;
  meanLum: number;
  /** 16×9 grid of cell-mean luminances, row-major. */
  grid: number[];
}

export async function captureGolden(renderer: Renderer, name: string): Promise<GoldenEntry> {
  const tp = renderer.tracePass;
  tp.setRefinementEnabled(false);
  try {
    for (let f = 0; f < 600; f++) {
      renderer.renderFrame();
      await renderer.onGpuIdle();
      await new Promise((r) => setTimeout(r, 0));
      if (tp.getKnownMarching() === 0) break;
    }
    const { data, width, height } = await renderer.readHdr();
    const GX = 16;
    const GY = 9;
    const grid = new Array<number>(GX * GY).fill(0);
    const counts = new Array<number>(GX * GY).fill(0);
    let sum = 0;
    for (let y = 0; y < height; y++) {
      const gy = Math.min(GY - 1, Math.floor((y * GY) / height));
      for (let x = 0; x < width; x++) {
        const px = (y * width + x) * 4;
        const lum = 0.2126 * data[px]! + 0.7152 * data[px + 1]! + 0.0722 * data[px + 2]!;
        const gx = Math.min(GX - 1, Math.floor((x * GX) / width));
        grid[gy * GX + gx] = grid[gy * GX + gx]! + lum;
        counts[gy * GX + gx] = counts[gy * GX + gx]! + 1;
        sum += lum;
      }
    }
    for (let c = 0; c < GX * GY; c++) grid[c] = grid[c]! / Math.max(1, counts[c]!);
    return { name, meanLum: sum / (width * height), grid };
  } finally {
    tp.setRefinementEnabled(true);
  }
}

// ---------------------------------------------------------------------------

export interface BoostedVOReport {
  spin: number;
  r: number;
  /** Orbital velocity of the observer measured by the local ZAMO (aberration scale). */
  zamoRelativeSpeed: number;
  /** Max aberration angle between boosted and ZAMO frame directions, degrees. */
  maxAberrationDeg: number;
  rows: VORow[];
  maxArcmin: number;
  statusAgreement: boolean;
  pass: boolean;
}

/**
 * M5 observer spot-check (IMPLEMENTATION.md M5): V-O1 rerun with a BOOSTED
 * tetrad — rays cast through the circular-orbit observer frame at r = 10.
 * Aberration lives entirely in the choice of u (MASTER_PLAN.md §5); this
 * confronts the f32 GPU march with the f64 oracle on rays whose initial
 * momenta carry the full orbital boost (v ≈ 0.35c at r = 10, a = 0), and
 * separately reports how large the aberration actually is relative to the
 * ZAMO frame, so a silently-ignored boost cannot pass.
 */
export async function runVOBoostedSuite(tracePass: TracePass, spin = 0, r = 10): Promise<BoostedVOReport> {
  const theta = Math.PI / 2;
  const m = createMetric();
  evalMetric(spin, r, theta, m);
  const uCirc = circularFourVelocity(spin, r);
  const boosted = orthonormalTetrad(m, uCirc);
  const zamo = orthonormalTetrad(m, zamoFourVelocity(m, spin, r));

  // Observer speed relative to the local ZAMO: cosh(rapidity) = -u·u_zamo.
  const uZamo = zamoFourVelocity(m, spin, r);
  const gamma = -innerProductBL(m, uCirc, uZamo);
  const zamoRelativeSpeed = Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));

  // Local directions spanning forward/backward/transverse, including
  // horizon-grazing offsets (the aberration-sensitive part of the sky).
  const dirs: Array<[number, number, number]> = [
    [-1, 0, 0],
    [-1, 0.25, 0],
    [-1, -0.25, 0.15],
    [-1, 0, 0.45],
    [-1, 0, -0.45],
    [1, 0.3, 0.3],
    [0.2, 0.1, -1],
  ];

  let maxAberrationDeg = 0;
  const rays: Array<{ label: string; state: Float64Array }> = dirs.map((n) => {
    // Aberration magnitude: how far the same local n̂ tilts when the boosted
    // frame's photon is re-expressed in the ZAMO frame.
    const kB = photonFromTetrad(m, boosted, n, { r, theta });
    const cosAb = aberrationCos(m, boosted, zamo, n);
    maxAberrationDeg = Math.max(maxAberrationDeg, (Math.acos(Math.min(1, Math.max(-1, cosAb))) * 180) / Math.PI);
    return { label: `boosted n=(${n.join(',')})`, state: kB };
  });

  const batch = new Float64Array(rays.length * STATE_SIZE);
  rays.forEach((ray, i) => batch.set(ray.state, i * STATE_SIZE));
  const quantized = Float64Array.from(batch, Math.fround);
  const gpu = await tracePass.runBatch(batch, spin);

  const scratch = createRhsScratch();
  const rows: VORow[] = [];
  let maxArcmin = 0;
  let statusAgreement = true;
  rays.forEach((ray, i) => {
    const y0 = quantized.subarray(i * STATE_SIZE, (i + 1) * STATE_SIZE);
    const oracle = integrateGeodesic(spin, Float64Array.from(y0), { rtol: 1e-13, atol: 1e-15 });
    const gpuStatus = gpu.statuses[i]!;
    const gpuFinal = gpu.states.subarray(i * STATE_SIZE, (i + 1) * STATE_SIZE) as Float64Array;
    const oracleName = oracle.status === 'CAPTURED' ? 'CAPTURED' : 'ESCAPED';
    const gpuName =
      gpuStatus === STATUS_CAPTURED ? 'CAPTURED' : gpuStatus === STATUS_ESCAPED ? 'ESCAPED' : `code ${gpuStatus}`;
    const statusAgrees = oracleName === gpuName;
    if (!statusAgrees) statusAgreement = false;
    let arcmin: number | null = null;
    if (oracle.status === 'ESCAPED' && gpuStatus === STATUS_ESCAPED && oracle.escapeDirection) {
      arcmin = angularSeparationArcmin(gpu.directions.subarray(i * 3, i * 3 + 3), oracle.escapeDirection);
      maxArcmin = Math.max(maxArcmin, arcmin);
    }
    rows.push({
      label: ray.label,
      bOffset: null,
      oracleStatus: oracleName,
      gpuStatus: gpuName,
      statusAgrees,
      arcmin,
      gpuSteps: gpu.steps[i]!,
      driftQ: Math.abs(carterConstant(spin, gpuFinal) - carterConstant(spin, Float64Array.from(y0))),
      residualH: hamiltonianResidual(spin, gpuFinal, scratch),
    });
  });

  return {
    spin,
    r,
    zamoRelativeSpeed,
    maxAberrationDeg,
    rows,
    maxArcmin,
    statusAgreement,
    // The boost must be physically present (aberration of order v) AND the
    // GPU must agree with the oracle to the V-O1 high-tier bound.
    pass: statusAgreement && maxArcmin <= ARCMIN_BOUND_HIGH && maxAberrationDeg > (zamoRelativeSpeed * 180) / Math.PI / 4,
  };
}

/** Metric inner product of two contravariant vectors (local helper). */
function innerProductBL(
  m: ReturnType<typeof createMetric>,
  u: ArrayLike<number>,
  v: ArrayLike<number>,
): number {
  return (
    m.g_tt * u[0]! * v[0]! +
    m.g_tph * (u[0]! * v[3]! + u[3]! * v[0]!) +
    m.g_rr * u[1]! * v[1]! +
    m.g_thth * u[2]! * v[2]! +
    m.g_phph * u[3]! * v[3]!
  );
}

/**
 * Cosine of the angle between the sky directions that the same coordinate
 * photon direction n̂-in-frame-A subtends when re-expressed in frame B:
 * build k from frame A, project onto frame B's spatial legs, compare with n̂.
 */
function aberrationCos(
  m: ReturnType<typeof createMetric>,
  frameA: Tetrad,
  frameB: Tetrad,
  n: [number, number, number],
): number {
  const norm = Math.hypot(n[0], n[1], n[2]);
  const nn = [n[0] / norm, n[1] / norm, n[2] / norm];
  const k: [number, number, number, number] = [0, 0, 0, 0];
  for (let mu = 0; mu < 4; mu++) {
    k[mu] = frameA[0][mu]! + nn[0]! * frameA[1][mu]! + nn[1]! * frameA[2][mu]! + nn[2]! * frameA[3][mu]!;
  }
  // Project k onto frame B: local components are η^{ab} g(k, e_b).
  const comps = [0, 1, 2, 3].map((aIdx) => {
    const eta = aIdx === 0 ? -1 : 1;
    return eta * innerProductBL(m, k, frameB[aIdx]!);
  });
  const spatial = Math.hypot(comps[1]!, comps[2]!, comps[3]!);
  return (comps[1]! * nn[0]! + comps[2]! * nn[1]! + comps[3]! * nn[2]!) / spatial;
}

// ---------------------------------------------------------------------------
// M8 — V-I5 golden comparison (VALIDATION.md §3): perceptual SSIM on the
// displayed (post-tonemap, arriving-photon-parity) image AND physical
// pre-tonemap HDR luminance statistics, both against committed baselines in
// tests/golden/. The whole measurement runs inside one refinement-off window
// so every readback sees the converged base pixel-center sample
// (ARCHITECTURE.md §5 contract; same discipline as captureGolden).
// ---------------------------------------------------------------------------

/** Preset name → committed golden file slug (tests/golden/<slug>.png). */
export function presetSlug(name: string): string {
  return name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
}

/** Browser-side PNG → grayscale luma decode (goldens are alpha-255 sRGB PNGs;
 * decode through 2D canvas is lossless for them). */
export async function decodePngToLuma(
  blob: Blob,
): Promise<{ data: Float64Array; width: number; height: number }> {
  const bmp = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('golden decode: no 2d context');
  ctx.drawImage(bmp, 0, 0);
  const img = ctx.getImageData(0, 0, bmp.width, bmp.height);
  const bytes = new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.length);
  return { data: rgba8ToLuma(bytes, bmp.width, bmp.height), width: bmp.width, height: bmp.height };
}

async function convergeBaseSample(renderer: Renderer, maxFrames = 600): Promise<number> {
  let frames = 0;
  for (; frames < maxFrames; frames++) {
    renderer.renderFrame();
    await renderer.onGpuIdle();
    await new Promise((r) => setTimeout(r, 0)); // let the counter mapAsync settle
    if (renderer.tracePass.getKnownMarching() === 0) break;
  }
  return frames;
}

/** Full-frame mean + 16×9 cell-mean pre-tonemap luminance (the captureGolden
 * statistics, computed from an already-read HDR frame). */
function goldenStats(data: Float32Array, width: number, height: number): { meanLum: number; grid: number[] } {
  const GX = 16;
  const GY = 9;
  const grid = new Array<number>(GX * GY).fill(0);
  const counts = new Array<number>(GX * GY).fill(0);
  let sum = 0;
  for (let y = 0; y < height; y++) {
    const gy = Math.min(GY - 1, Math.floor((y * GY) / height));
    for (let x = 0; x < width; x++) {
      const px = (y * width + x) * 4;
      const lum = 0.2126 * data[px]! + 0.7152 * data[px + 1]! + 0.0722 * data[px + 2]!;
      const gx = Math.min(GX - 1, Math.floor((x * GX) / width));
      grid[gy * GX + gx] = grid[gy * GX + gx]! + lum;
      counts[gy * GX + gx] = counts[gy * GX + gx]! + 1;
      sum += lum;
    }
  }
  for (let c = 0; c < GX * GY; c++) grid[c] = grid[c]! / Math.max(1, counts[c]!);
  return { meanLum: sum / (width * height), grid };
}

/** Converged displayed-image capture for golden baselining (rgba8, present
 * transform applied — the exact pixels a user sees/screenshots). */
export async function captureGoldenImage(
  renderer: Renderer,
): Promise<{ bytes: Uint8Array; width: number; height: number; framesToConverge: number }> {
  const tp = renderer.tracePass;
  tp.setRefinementEnabled(false);
  try {
    const frames = await convergeBaseSample(renderer);
    const { bytes, width, height } = await renderer.presentToBytes();
    return { bytes, width, height, framesToConverge: frames };
  } finally {
    tp.setRefinementEnabled(true);
  }
}

export interface GoldenCompareRow {
  name: string;
  /** |mean_now / mean_baseline − 1| of pre-tonemap luminance (bound 1 %). */
  meanLumRelErr: number;
  /** Max cell-mean relative error over cells with baseline mean > 2 % of the
   * frame mean (dim cells are quantization-dominated); reported, not gated. */
  maxCellRelErr: number;
  /** Mean SSIM of displayed luma vs the committed golden (bound > 0.99). */
  ssim: number;
  ssimMin: number;
  ssimMinX: number;
  ssimMinY: number;
  framesToConverge: number;
  pass: boolean;
}

/**
 * V-I5 comparison for one preset (the caller applies the preset and matches
 * the baseline resolution first): SSIM > 0.99 on the displayed image AND
 * mean pre-tonemap HDR luminance error < 1 % (VALIDATION.md §3).
 */
export async function compareGolden(
  renderer: Renderer,
  baseline: GoldenEntry,
  goldenLuma: { data: Float64Array; width: number; height: number },
): Promise<GoldenCompareRow> {
  const tp = renderer.tracePass;
  tp.setRefinementEnabled(false);
  try {
    const frames = await convergeBaseSample(renderer);
    const { data, width, height } = await renderer.readHdr();
    if (width !== goldenLuma.width || height !== goldenLuma.height) {
      throw new Error(
        `golden ${baseline.name}: resolution ${width}×${height} does not match baseline ${goldenLuma.width}×${goldenLuma.height}`,
      );
    }
    const stats = goldenStats(data, width, height);
    const meanLumRelErr = Math.abs(stats.meanLum / baseline.meanLum - 1);
    let maxCellRelErr = 0;
    for (let c = 0; c < baseline.grid.length; c++) {
      const base = baseline.grid[c]!;
      if (base <= 0.02 * baseline.meanLum) continue;
      maxCellRelErr = Math.max(maxCellRelErr, Math.abs(stats.grid[c]! / base - 1));
    }

    const { bytes } = await renderer.presentToBytes();
    const ssim = computeSsim(rgba8ToLuma(bytes, width, height), goldenLuma.data, width, height);

    return {
      name: baseline.name,
      meanLumRelErr,
      maxCellRelErr,
      ssim: ssim.mean,
      ssimMin: ssim.min,
      ssimMinX: ssim.minX,
      ssimMinY: ssim.minY,
      framesToConverge: frames,
      pass: meanLumRelErr < 0.01 && ssim.mean > 0.99,
    };
  } finally {
    tp.setRefinementEnabled(true);
  }
}

// ---------------------------------------------------------------------------
// M8 — V-O3 (VALIDATION.md §2): fraction of screen rays flagged MAX_STEPS at
// convergence, per shipped preset. Bound: < 0.1 % of pixels, all presets.
// ---------------------------------------------------------------------------

export interface StatusCensusReport {
  name: string;
  width: number;
  height: number;
  pixels: number;
  marching: number;
  captured: number;
  escaped: number;
  hitDisk: number;
  maxSteps: number;
  maxStepsFraction: number;
  framesToConverge: number;
  pass: boolean;
}

export async function measureStatusCensus(renderer: Renderer, name: string): Promise<StatusCensusReport> {
  const tp = renderer.tracePass;
  tp.setRefinementEnabled(false);
  try {
    const frames = await convergeBaseSample(renderer);
    const statuses = await tp.readScreenStatuses();
    const { width, height } = renderer.dimensions();
    let marching = 0;
    let captured = 0;
    let escaped = 0;
    let hitDisk = 0;
    let maxSteps = 0;
    for (let i = 0; i < width * height; i++) {
      const s = statuses[i]!;
      if (s === STATUS_CAPTURED) captured++;
      else if (s === STATUS_ESCAPED) escaped++;
      else if (s === STATUS_HIT_DISK) hitDisk++;
      else if (s === STATUS_MAX_STEPS) maxSteps++;
      else marching++;
    }
    const maxStepsFraction = maxSteps / (width * height);
    return {
      name,
      width,
      height,
      pixels: width * height,
      marching,
      captured,
      escaped,
      hitDisk,
      maxSteps,
      maxStepsFraction,
      framesToConverge: frames,
      pass: marching === 0 && maxStepsFraction < 1e-3,
    };
  } finally {
    tp.setRefinementEnabled(true);
  }
}

// ---------------------------------------------------------------------------
// M8 — V-I2 (VALIDATION.md §3): disk-image boundary geometry at the Luminet
// framing. Along radial spokes from the frame center, every strict
// transition of the GPU status field into/out of HIT_DISK (primary image
// outer/inner edges, the secondary arcs over and under the shadow) is
// bracketed at pixel resolution and confronted with the float64 oracle:
// the same bracket is bisected on the oracle's disk-hit predicate to
// sub-pixel radius. The pass metric is the relative radial deviation of
// every matched boundary point. Luminet's published figure is itself the
// numerical solution of these equations for this geometry [R2] — the
// float64 oracle (validated against closed forms in V-A) stands in for the
// published curves at full precision; the rendered-image flux comparison on
// the same framing is measureBeamingAsymmetry (V-I3 oracle half).
// ---------------------------------------------------------------------------

export interface DiskEdgeRow {
  /** Spoke angle in image plane, radians (screen +x toward 0, +y down). */
  angle: number;
  rhoGpu: number;
  rhoOracle: number;
  devPx: number;
  relDev: number;
  /** true if the ray enters the disk image at this radius (miss→hit). */
  entering: boolean;
}

export interface DiskEdgeReport {
  spin: number;
  width: number;
  height: number;
  spokes: number;
  transitions: number;
  matched: number;
  /** Brackets where the oracle's membership at the endpoints disagrees with
   * the GPU samples (status-fragile edge pixels; reported, cf. V-O2). */
  fragile: number;
  maxDevPx: number;
  meanDevPx: number;
  /** Max radial deviation relative to the local boundary radius, all
   * transitions — quantization-limited below ρ ≈ 50 px (the GPU boundary is
   * only known to the half-pixel midpoint, ±0.7 px ⇒ > 2 % of a small ρ), so
   * this is reported, not bounded. */
  maxRelDev: number;
  /** The bounded shape metric: max relative deviation over transitions with
   * ρ ≥ 50 px, where the 2 % criterion is resolvable above quantization. */
  maxRelDevBounded: number;
  rows: DiskEdgeRow[];
  framesToConverge: number;
  pass: boolean;
}

export async function measureDiskEdgeGeometry(
  renderer: Renderer,
  camera: OrbitCamera,
  spin: number,
): Promise<DiskEdgeReport> {
  const tp = renderer.tracePass;
  const luts = tp.getLuts();
  if (!luts) throw new Error('disk LUTs not loaded');
  const savedDisk = tp.getDisk();
  if (!savedDisk.enabled) tp.setDisk({ ...savedDisk, enabled: true });
  tp.setRefinementEnabled(false);
  try {
    camera.set({ ...VI3_FRAMING });
    const frames = await convergeBaseSample(renderer, 400);

    const statuses = await tp.readScreenStatuses();
    const { width, height } = renderer.dimensions();
    const shading = tp.getDiskShading();
    const pose = tp.getCamera();
    if (!shading || !pose) throw new Error('disk shading state unavailable');

    const m = createMetric();
    evalMetric(spin, pose.r, pose.theta, m);
    const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, spin, pose.r));
    const tanY = Math.tan(pose.fovY / 2);
    const tanX = (tanY * width) / height;
    const disk = { rInner: shading.rIsco, rOuter: shading.rOuter };

    /** Oracle disk-hit predicate at a continuous pixel position. */
    const oracleHits = (px: number, py: number): boolean => {
      const ndcX = (2 * (px + 0.5)) / width - 1;
      const ndcY = 1 - (2 * (py + 0.5)) / height;
      const n: [number, number, number] = [-1, -ndcY * tanY, ndcX * tanX];
      const len = Math.hypot(...n);
      const y0 = photonFromTetrad(
        m,
        tetrad,
        [n[0] / len, n[1] / len, n[2] / len],
        { r: pose.r, theta: pose.theta, phi: pose.phi },
      );
      return integrateGeodesic(spin, y0, { rtol: 1e-12, atol: 1e-14, disk }).status === 'HIT_DISK';
    };

    const cx = width / 2 - 0.5;
    const cy = height / 2 - 0.5;

    const SPOKES = 48;
    const rows: DiskEdgeRow[] = [];
    let transitions = 0;
    let fragile = 0;
    let maxDevPx = 0;
    let sumDevPx = 0;
    let maxRelDev = 0;
    let maxRelDevBounded = 0;
    for (let k = 0; k < SPOKES; k++) {
      const angle = (2 * Math.PI * k) / SPOKES;
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      // Walk the spoke in ~1 px steps, but SAMPLE at the pixel centers the
      // GPU actually shaded: both implementations are then evaluated at the
      // identical screen positions, so an endpoint disagreement is a genuine
      // f32↔f64 status flip (fragile edge pixel), not a rounding artifact.
      let prev: { x: number; y: number; member: boolean } | null = null;
      for (let rho = 2; ; rho++) {
        const x = Math.round(cx + rho * dx);
        const y = Math.round(cy + rho * dy);
        if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) break;
        if (prev && x === prev.x && y === prev.y) continue; // aliased step
        const member: boolean = statuses[y * width + x] === STATUS_HIT_DISK;
        if (prev && member !== prev.member) {
          transitions++;
          if (oracleHits(prev.x, prev.y) !== prev.member || oracleHits(x, y) !== member) {
            fragile++;
          } else {
            // Bisect the oracle's boundary along the segment between the two
            // pixel centers; radii are distances to the spoke origin.
            let ta = 0;
            let tb = 1;
            const seg = Math.hypot(x - prev.x, y - prev.y);
            while ((tb - ta) * seg > 0.2) {
              const tm = 0.5 * (ta + tb);
              const hit = oracleHits(prev.x + tm * (x - prev.x), prev.y + tm * (y - prev.y));
              if (hit === prev.member) ta = tm;
              else tb = tm;
            }
            const t = 0.5 * (ta + tb);
            const bx = prev.x + t * (x - prev.x);
            const by = prev.y + t * (y - prev.y);
            const rhoOracle = Math.hypot(bx - cx, by - cy);
            const rhoGpu = Math.hypot(0.5 * (x + prev.x) - cx, 0.5 * (y + prev.y) - cy);
            const devPx = Math.abs(rhoGpu - rhoOracle);
            const relDev = devPx / rhoOracle;
            maxDevPx = Math.max(maxDevPx, devPx);
            sumDevPx += devPx;
            maxRelDev = Math.max(maxRelDev, relDev);
            if (rhoOracle >= 50) maxRelDevBounded = Math.max(maxRelDevBounded, relDev);
            rows.push({ angle, rhoGpu, rhoOracle, devPx, relDev, entering: member });
          }
        }
        prev = { x, y, member };
      }
    }

    const matched = rows.length;
    return {
      spin,
      width,
      height,
      spokes: SPOKES,
      transitions,
      matched,
      fragile,
      maxDevPx,
      meanDevPx: matched > 0 ? sumDevPx / matched : Infinity,
      maxRelDev,
      maxRelDevBounded,
      rows,
      framesToConverge: frames,
      // Shape within 2 % where resolvable, and never more than one pixel of
      // absolute boundary deviation anywhere (the V-I1 tolerance philosophy).
      pass: transitions >= 40 && matched >= 0.9 * transitions && maxDevPx <= 1.0 && maxRelDevBounded < 0.02,
    };
  } finally {
    tp.setRefinementEnabled(true);
    tp.setDisk(savedDisk);
  }
}
