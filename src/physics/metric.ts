/**
 * PHYSICS.md §2 — Kerr metric in Boyer-Lindquist coordinates (D4), G = c = M = 1.
 *
 * The geodesic engine needs only the inverse metric and its analytic r/θ
 * gradients (Hamiltonian formulation, D3). Covariant components are provided
 * for tetrads and index lowering. Gradients are hand-derived quotient rules;
 * they are verified against Richardson finite differences in the test suite,
 * and the WGSL port must be a literal translation of this file.
 *
 * Notation matches the docs: Σ = r² + a²cos²θ, Δ = r² − 2r + a²,
 * A = (r² + a²)² − a²Δsin²θ.
 */

/** Outer event horizon r₊ = 1 + √(1 − a²). */
export function horizonRadius(a: number): number {
  return 1 + Math.sqrt(1 - a * a);
}

/** Ergosphere boundary r_E(θ) = 1 + √(1 − a²cos²θ). */
export function ergosphereRadius(a: number, theta: number): number {
  const c = Math.cos(theta);
  return 1 + Math.sqrt(1 - a * a * c * c);
}

export interface MetricBL {
  /** covariant */
  g_tt: number;
  g_tph: number;
  g_rr: number;
  g_thth: number;
  g_phph: number;
  /** contravariant */
  gtt: number;
  gtph: number;
  grr: number;
  gthth: number;
  gphph: number;
}

export interface InverseMetricGradients {
  /** ∂/∂r of the contravariant components */
  dr_gtt: number;
  dr_gtph: number;
  dr_grr: number;
  dr_gthth: number;
  dr_gphph: number;
  /** ∂/∂θ of the contravariant components */
  dth_gtt: number;
  dth_gtph: number;
  dth_grr: number;
  dth_gthth: number;
  dth_gphph: number;
}

export function createMetric(): MetricBL {
  return {
    g_tt: 0, g_tph: 0, g_rr: 0, g_thth: 0, g_phph: 0,
    gtt: 0, gtph: 0, grr: 0, gthth: 0, gphph: 0,
  };
}

export function createGradients(): InverseMetricGradients {
  return {
    dr_gtt: 0, dr_gtph: 0, dr_grr: 0, dr_gthth: 0, dr_gphph: 0,
    dth_gtt: 0, dth_gtph: 0, dth_grr: 0, dth_gthth: 0, dth_gphph: 0,
  };
}

/** Evaluate covariant + contravariant Kerr components at (r, θ) into `out`. */
export function evalMetric(a: number, r: number, theta: number, out: MetricBL): MetricBL {
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  const s2 = s * s;
  const r2 = r * r;
  const a2 = a * a;

  const Sigma = r2 + a2 * c * c;
  const Delta = r2 - 2 * r + a2;
  const A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;

  out.g_tt = -1 + (2 * r) / Sigma;
  out.g_tph = (-2 * a * r * s2) / Sigma;
  out.g_rr = Sigma / Delta;
  out.g_thth = Sigma;
  out.g_phph = (A * s2) / Sigma;

  const SigmaDelta = Sigma * Delta;
  out.gtt = -A / SigmaDelta;
  out.gtph = (-2 * a * r) / SigmaDelta;
  out.grr = Delta / Sigma;
  out.gthth = 1 / Sigma;
  out.gphph = (Delta - a2 * s2) / (SigmaDelta * s2);
  return out;
}

/** Analytic r/θ gradients of the contravariant components at (r, θ) into `out`. */
export function evalInverseMetricGradients(
  a: number,
  r: number,
  theta: number,
  out: InverseMetricGradients,
): InverseMetricGradients {
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  const s2 = s * s;
  const r2 = r * r;
  const a2 = a * a;

  const Sigma = r2 + a2 * c * c;
  const Delta = r2 - 2 * r + a2;
  const A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;

  const Sigma_r = 2 * r;
  const Sigma_th = -2 * a2 * s * c;
  const Delta_r = 2 * r - 2;
  const A_r = 4 * r * (r2 + a2) - a2 * Delta_r * s2;
  const A_th = -2 * a2 * Delta * s * c;

  const SD = Sigma * Delta;
  const SD2 = SD * SD;
  const SD_r = Sigma_r * Delta + Sigma * Delta_r;
  const SD_th = Sigma_th * Delta;

  // g^tt = −A/(ΣΔ)
  out.dr_gtt = -(A_r * SD - A * SD_r) / SD2;
  out.dth_gtt = -(A_th * SD - A * SD_th) / SD2;

  // g^tφ = −2ar/(ΣΔ)
  out.dr_gtph = (-2 * a * (SD - r * SD_r)) / SD2;
  out.dth_gtph = (2 * a * r * SD_th) / SD2;

  // g^rr = Δ/Σ
  out.dr_grr = (Delta_r * Sigma - Delta * Sigma_r) / (Sigma * Sigma);
  out.dth_grr = (-Delta * Sigma_th) / (Sigma * Sigma);

  // g^θθ = 1/Σ
  out.dr_gthth = -Sigma_r / (Sigma * Sigma);
  out.dth_gthth = -Sigma_th / (Sigma * Sigma);

  // g^φφ = (Δ − a²sin²θ)/(ΣΔ sin²θ)
  const N = Delta - a2 * s2;
  const D = SD * s2;
  const D2 = D * D;
  const N_r = Delta_r;
  const D_r = SD_r * s2;
  const N_th = -2 * a2 * s * c;
  const D_th = Sigma_th * Delta * s2 + SD * 2 * s * c;
  out.dr_gphph = (N_r * D - N * D_r) / D2;
  out.dth_gphph = (N_th * D - N * D_th) / D2;

  return out;
}

/**
 * Inner product g(u, v) of two contravariant vectors in BL component order
 * (t, r, θ, φ), using covariant components of `m`.
 */
export function innerProduct(
  m: MetricBL,
  u: readonly number[],
  v: readonly number[],
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
 * BL → pseudo-Cartesian direction of motion (unit 3-vector) at a phase-space
 * point, using x = √(r²+a²) sinθcosφ, y = √(r²+a²) sinθ sinφ, z = r cosθ.
 * Used for asymptotic sky directions of escaped rays (PHYSICS.md §10).
 */
export function cartesianDirection(
  a: number,
  r: number,
  theta: number,
  phi: number,
  dr: number,
  dtheta: number,
  dphi: number,
): [number, number, number] {
  const R = Math.sqrt(r * r + a * a);
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  const cp = Math.cos(phi);
  const sp = Math.sin(phi);
  const dR = (r / R) * dr;
  const dx = dR * s * cp + R * c * cp * dtheta - R * s * sp * dphi;
  const dy = dR * s * sp + R * c * sp * dtheta + R * s * cp * dphi;
  const dz = c * dr - r * s * dtheta;
  const n = Math.hypot(dx, dy, dz);
  return [dx / n, dy / n, dz / n];
}
