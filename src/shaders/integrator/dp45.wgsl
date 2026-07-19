// Dormand-Prince RK5(4) geodesic step — float32 port of
// src/physics/integrator.ts (same tableau, error norm, PI controller, and
// horizon clamp; PHYSICS.md §10). In-kernel conserved-quantity monitors are
// deferred to M6 tuning — V-O4 measures drift externally from readback.

#include "../metric/kerr.wgsl"

// Dormand-Prince tableau [R12].
const A21: f32 = 0.2;
const A31: f32 = 0.075;
const A32: f32 = 0.225;
const A41 = 44.0 / 45.0;
const A42 = -56.0 / 15.0;
const A43 = 32.0 / 9.0;
const A51 = 19372.0 / 6561.0;
const A52 = -25360.0 / 2187.0;
const A53 = 64448.0 / 6561.0;
const A54 = -212.0 / 729.0;
const A61 = 9017.0 / 3168.0;
const A62 = -355.0 / 33.0;
const A63 = 46732.0 / 5247.0;
const A64 = 49.0 / 176.0;
const A65 = -5103.0 / 18656.0;
const B1 = 35.0 / 384.0;
const B3 = 500.0 / 1113.0;
const B4 = 125.0 / 192.0;
const B5 = -2187.0 / 6784.0;
const B6 = 11.0 / 84.0;
const E1 = B1 - 5179.0 / 57600.0;
const E3 = B3 - 7571.0 / 16695.0;
const E4 = B4 - 393.0 / 640.0;
const E5 = B5 + 92097.0 / 339200.0;
const E6 = B6 - 187.0 / 2100.0;
const E7 = -1.0 / 40.0;

const SAFETY: f32 = 0.9;
const MIN_SCALE: f32 = 0.2;
const MAX_SCALE: f32 = 5.0;
const PI_ALPHA: f32 = 0.14; // 0.7/5
const PI_BETA: f32 = 0.08;  // 0.4/5

// Geodesic RHS: dx = g^{..} p ; dp = -1/2 (d g^{..}) p p  (PHYSICS.md §4).
// p_t and p_phi RHS are identically zero (exact E, L_z conservation).
fn geodesicRhs(a: f32, y: array<f32, 8>) -> array<f32, 8> {
  let r = y[IR];
  let th = y[ITH];
  let pt = y[IPT];
  let pr = y[IPR];
  let pth = y[IPTH];
  let pph = y[IPPH];

  let m = evalInverseMetric(a, r, th);
  let g = evalInverseMetricGradients(a, r, th);

  var f: array<f32, 8>;
  f[IT] = m.gtt * pt + m.gtph * pph;
  f[IR] = m.grr * pr;
  f[ITH] = m.gthth * pth;
  f[IPT] = 0.0;
  f[IPPH] = 0.0;

  // phi-phi channel through l = p_phi/sin(theta) — literal port of the
  // axis-regular formulation in src/physics/hamiltonian.ts: p_phi = 0 rays
  // cross the polar axis, where their g^phiphi-coupled terms vanish
  // identically (skipping evaluates the exact limit instead of Inf*0 = NaN).
  var dphi = m.gtph * pt;
  var phiTermR = 0.0;
  var phiTermTh = 0.0;
  if (pph != 0.0) {
    let s = sin(th);
    let lOverS = pph / s;
    dphi += m.gphphS2 * (lOverS / s);
    phiTermR = g.dr_gphphS2 * lOverS * lOverS;
    phiTermTh = (g.dth_gphphS3 * lOverS * lOverS) / s;
  }
  f[IPH] = dphi;

  f[IPR] = -0.5 * (
    g.dr_gtt * pt * pt +
    2.0 * g.dr_gtph * pt * pph +
    g.dr_grr * pr * pr +
    g.dr_gthth * pth * pth +
    phiTermR
  );
  f[IPTH] = -0.5 * (
    g.dth_gtt * pt * pt +
    2.0 * g.dth_gtph * pt * pph +
    g.dth_grr * pr * pr +
    g.dth_gthth * pth * pth +
    phiTermTh
  );
  return f;
}

// Initial step size from RHS scale (port of the oracle heuristic).
fn initialStepSize(y: array<f32, 8>, k1: array<f32, 8>, rtol: f32, atol: f32) -> f32 {
  var h = 1e30;
  for (var i = 0; i < 8; i++) {
    let sc = atol + rtol * abs(y[i]);
    let fi = abs(k1[i]);
    if (fi > 0.0) {
      h = min(h, sc / fi);
    }
  }
  return min(max(h * 1e3, 1e-8), 1.0);
}

struct StepAttempt {
  yTrial: array<f32, 8>,
  k7: array<f32, 8>,
  err: f32,
}

// Cubic Hermite interpolation of the full state across an accepted step,
// sigma in [0, 1] — port of hermiteState in src/physics/integrator.ts.
fn hermiteState(
  y0: array<f32, 8>,
  f0: array<f32, 8>,
  y1: array<f32, 8>,
  f1: array<f32, 8>,
  h: f32,
  sigma: f32,
) -> array<f32, 8> {
  let s2 = sigma * sigma;
  let s3 = s2 * sigma;
  let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
  let h10 = s3 - 2.0 * s2 + sigma;
  let h01 = -2.0 * s3 + 3.0 * s2;
  let h11 = s3 - s2;
  var out: array<f32, 8>;
  for (var i = 0; i < 8; i++) {
    out[i] = h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i];
  }
  return out;
}

// Hermite interpolation of the theta component only — port of
// hermiteComponent(…, ITH) in src/physics/integrator.ts (used by the disk
// plane-crossing bisection, where interpolating the full state per iteration
// would be wasted work).
fn hermiteTheta(
  y0: array<f32, 8>,
  f0: array<f32, 8>,
  y1: array<f32, 8>,
  f1: array<f32, 8>,
  h: f32,
  sigma: f32,
) -> f32 {
  let s2 = sigma * sigma;
  let s3 = s2 * sigma;
  return (2.0 * s3 - 3.0 * s2 + 1.0) * y0[ITH]
    + (s3 - 2.0 * s2 + sigma) * h * f0[ITH]
    + (-2.0 * s3 + 3.0 * s2) * y1[ITH]
    + (s3 - s2) * h * f1[ITH];
}

// One DP45 stage evaluation sweep at step size h from (y, k1).
fn dp45Attempt(a: f32, y: array<f32, 8>, k1: array<f32, 8>, h: f32, rtol: f32, atol: f32) -> StepAttempt {
  var ys: array<f32, 8>;

  for (var i = 0; i < 8; i++) { ys[i] = y[i] + h * A21 * k1[i]; }
  let k2 = geodesicRhs(a, ys);
  for (var i = 0; i < 8; i++) { ys[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]); }
  let k3 = geodesicRhs(a, ys);
  for (var i = 0; i < 8; i++) { ys[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]); }
  let k4 = geodesicRhs(a, ys);
  for (var i = 0; i < 8; i++) { ys[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]); }
  let k5 = geodesicRhs(a, ys);
  for (var i = 0; i < 8; i++) { ys[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]); }
  let k6 = geodesicRhs(a, ys);

  var out: StepAttempt;
  for (var i = 0; i < 8; i++) {
    out.yTrial[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
  }
  out.k7 = geodesicRhs(a, out.yTrial);

  var errSq = 0.0;
  for (var i = 0; i < 8; i++) {
    let e = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * out.k7[i]);
    let sc = atol + rtol * max(abs(y[i]), abs(out.yTrial[i]));
    errSq += (e / sc) * (e / sc);
  }
  out.err = max(sqrt(errSq / 8.0), 1e-12);
  return out;
}
