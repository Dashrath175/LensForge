/**
 * PHYSICS.md §4 — Hamiltonian geodesic equations (D3).
 *
 * State layout (shared with the integrator and, in M2, the WGSL port):
 *   y = [t, r, θ, φ, p_t, p_r, p_θ, p_φ]
 *
 * dx^μ/dλ = g^{μν} p_ν ;  dp_μ/dλ = −½ (∂_μ g^{αβ}) p_α p_β.
 * p_t and p_φ are exact constants of motion (stationarity + axisymmetry), so
 * their RHS is identically zero; E and L_z therefore cannot drift and the
 * live error monitors are Q and H (PHYSICS.md §10).
 */
import {
  createGradients,
  createMetric,
  evalInverseMetricGradients,
  evalMetric,
  type InverseMetricGradients,
  type MetricBL,
} from './metric';

export const IT = 0;
export const IR = 1;
export const ITH = 2;
export const IPH = 3;
export const IPT = 4;
export const IPR = 5;
export const IPTH = 6;
export const IPPH = 7;
export const STATE_SIZE = 8;

/** Reusable scratch for allocation-free RHS evaluation in the integrator hot loop. */
export interface RhsScratch {
  metric: MetricBL;
  grad: InverseMetricGradients;
}

export function createRhsScratch(): RhsScratch {
  return { metric: createMetric(), grad: createGradients() };
}

/** Geodesic right-hand side dy/dλ at state `y`, written into `out`. */
export function geodesicRhs(
  a: number,
  y: Float64Array,
  out: Float64Array,
  scratch: RhsScratch,
): void {
  const r = y[IR]!;
  const th = y[ITH]!;
  const pt = y[IPT]!;
  const pr = y[IPR]!;
  const pth = y[IPTH]!;
  const pph = y[IPPH]!;

  const m = evalMetric(a, r, th, scratch.metric);
  const g = evalInverseMetricGradients(a, r, th, scratch.grad);

  out[IT] = m.gtt * pt + m.gtph * pph;
  out[IR] = m.grr * pr;
  out[ITH] = m.gthth * pth;
  out[IPH] = m.gtph * pt + m.gphph * pph;

  out[IPT] = 0;
  out[IPPH] = 0;
  out[IPR] = -0.5 * (
    g.dr_gtt * pt * pt +
    2 * g.dr_gtph * pt * pph +
    g.dr_grr * pr * pr +
    g.dr_gthth * pth * pth +
    g.dr_gphph * pph * pph
  );
  out[IPTH] = -0.5 * (
    g.dth_gtt * pt * pt +
    2 * g.dth_gtph * pt * pph +
    g.dth_grr * pr * pr +
    g.dth_gthth * pth * pth +
    g.dth_gphph * pph * pph
  );
}

/**
 * Well-conditioned null-constraint monitor: |H| = |½ g^{μν} p_μ p_ν|
 * normalized by the magnitude sum of its terms. Near the horizon (BL), H is
 * the difference of O(1/Δ) terms; normalizing by E² there would report
 * conditioning noise as drift.
 */
export function hamiltonianResidual(a: number, y: Float64Array, scratch: RhsScratch): number {
  const m = evalMetric(a, y[IR]!, y[ITH]!, scratch.metric);
  const pt = y[IPT]!;
  const pr = y[IPR]!;
  const pth = y[IPTH]!;
  const pph = y[IPPH]!;
  const tTerm = m.gtt * pt * pt;
  const crossTerm = 2 * m.gtph * pt * pph;
  const rTerm = m.grr * pr * pr;
  const thTerm = m.gthth * pth * pth;
  const phTerm = m.gphph * pph * pph;
  const scale =
    Math.abs(tTerm) + Math.abs(crossTerm) + Math.abs(rTerm) + Math.abs(thTerm) + Math.abs(phTerm);
  return Math.abs(tTerm + crossTerm + rTerm + thTerm + phTerm) / Math.max(scale, 1e-30);
}

/** Carter constant for photons: Q = p_θ² + cos²θ (L_z²/sin²θ − a²E²); error monitor. */
export function carterConstant(a: number, y: Float64Array): number {
  const th = y[ITH]!;
  const s = Math.sin(th);
  const c = Math.cos(th);
  const E = -y[IPT]!;
  const L = y[IPPH]!;
  const pth = y[IPTH]!;
  return pth * pth + c * c * ((L * L) / (s * s) - a * a * E * E);
}
