// Kerr metric, Boyer-Lindquist, G = c = M = 1 — float32 port of
// src/physics/metric.ts (literal translation; MASTER_PLAN.md §7).
// Δ uses the factored form (r − r₊)(r − r₋): exactly conditioned near the
// horizon where the polynomial form cancels catastrophically in f32.

struct InvMetric {
  gtt: f32,
  gtph: f32,
  grr: f32,
  gthth: f32,
  gphph: f32,
}

struct InvMetricGradients {
  dr_gtt: f32,
  dr_gtph: f32,
  dr_grr: f32,
  dr_gthth: f32,
  dr_gphph: f32,
  dth_gtt: f32,
  dth_gtph: f32,
  dth_grr: f32,
  dth_gthth: f32,
  dth_gphph: f32,
}

fn deltaFactored(a: f32, r: f32) -> f32 {
  let root = sqrt(1.0 - a * a);
  return (r - 1.0 - root) * (r - 1.0 + root);
}

fn evalInverseMetric(a: f32, r: f32, theta: f32) -> InvMetric {
  let s = sin(theta);
  let c = cos(theta);
  let s2 = s * s;
  let r2 = r * r;
  let a2 = a * a;

  let Sigma = r2 + a2 * c * c;
  let Delta = deltaFactored(a, r);
  let A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;
  let SigmaDelta = Sigma * Delta;

  var m: InvMetric;
  m.gtt = -A / SigmaDelta;
  m.gtph = (-2.0 * a * r) / SigmaDelta;
  m.grr = Delta / Sigma;
  m.gthth = 1.0 / Sigma;
  m.gphph = (Delta - a2 * s2) / (SigmaDelta * s2);
  return m;
}

fn evalInverseMetricGradients(a: f32, r: f32, theta: f32) -> InvMetricGradients {
  let s = sin(theta);
  let c = cos(theta);
  let s2 = s * s;
  let r2 = r * r;
  let a2 = a * a;

  let Sigma = r2 + a2 * c * c;
  let Delta = deltaFactored(a, r);
  let A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;

  let Sigma_r = 2.0 * r;
  let Sigma_th = -2.0 * a2 * s * c;
  let Delta_r = 2.0 * r - 2.0;
  let A_r = 4.0 * r * (r2 + a2) - a2 * Delta_r * s2;
  let A_th = -2.0 * a2 * Delta * s * c;

  let SD = Sigma * Delta;
  let SD2 = SD * SD;
  let SD_r = Sigma_r * Delta + Sigma * Delta_r;
  let SD_th = Sigma_th * Delta;

  var g: InvMetricGradients;
  g.dr_gtt = -(A_r * SD - A * SD_r) / SD2;
  g.dth_gtt = -(A_th * SD - A * SD_th) / SD2;

  g.dr_gtph = (-2.0 * a * (SD - r * SD_r)) / SD2;
  g.dth_gtph = (2.0 * a * r * SD_th) / SD2;

  g.dr_grr = (Delta_r * Sigma - Delta * Sigma_r) / (Sigma * Sigma);
  g.dth_grr = (-Delta * Sigma_th) / (Sigma * Sigma);

  g.dr_gthth = -Sigma_r / (Sigma * Sigma);
  g.dth_gthth = -Sigma_th / (Sigma * Sigma);

  let N = Delta - a2 * s2;
  let D = SD * s2;
  let D2 = D * D;
  let N_r = Delta_r;
  let D_r = SD_r * s2;
  let N_th = -2.0 * a2 * s * c;
  let D_th = Sigma_th * Delta * s2 + SD * 2.0 * s * c;
  g.dr_gphph = (N_r * D - N * D_r) / D2;
  g.dth_gphph = (N_th * D - N * D_th) / D2;

  return g;
}

// BL -> pseudo-Cartesian unit direction of motion (escaped-ray sky lookup),
// port of cartesianDirection in src/physics/metric.ts.
fn cartesianDirection(a: f32, r: f32, theta: f32, phi: f32, dr: f32, dtheta: f32, dphi: f32) -> vec3f {
  let R = sqrt(r * r + a * a);
  let s = sin(theta);
  let c = cos(theta);
  let cp = cos(phi);
  let sp = sin(phi);
  let dR = (r / R) * dr;
  let dx = dR * s * cp + R * c * cp * dtheta - R * s * sp * dphi;
  let dy = dR * s * sp + R * c * sp * dtheta + R * s * cp * dphi;
  let dz = c * dr - r * s * dtheta;
  return normalize(vec3f(dx, dy, dz));
}
