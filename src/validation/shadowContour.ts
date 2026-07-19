/**
 * V-I1 support (VALIDATION.md §3): exact mapping of the Bardeen shadow
 * boundary [R1] onto the screen of the renderer's camera.
 *
 * The critical curve is a family of conserved-quantity pairs (ξ = L_z/E,
 * η = Q/E²) — observer-independent (src/physics/shadow.ts). A traced camera
 * ray is captured iff its own (ξ, η) lies inside that family, at any camera
 * radius outside the photon shell. What depends on the observer is only the
 * map from screen direction to (ξ, η), and that map is linear in the tetrad
 * direction n̂, so it inverts exactly: no far-field approximation enters, and
 * finite-distance and ZAMO frame effects are included to float64 accuracy.
 *
 * Pure float64 (physics modules only) — shared by the Vitest contour test and
 * the browser V-I1 harness; keeps renderer imports out of the CPU test path.
 */
import { createMetric, evalMetric } from '../physics/metric';
import { bardeenContour } from '../physics/shadow';
import { orthonormalTetrad, zamoFourVelocity, type FourVector, type Tetrad } from '../physics/tetrad';
import type { MetricBL } from '../physics/metric';

export interface CameraFraming {
  r: number;
  theta: number;
  fovY: number;
  width: number;
  height: number;
}

/** Covariant components (t, θ, φ rows needed here) of g·e for a tetrad leg. */
function lower(m: MetricBL, e: FourVector): { t: number; th: number; ph: number } {
  return {
    t: m.g_tt * e[0] + m.g_tph * e[3],
    th: m.g_thth * e[2],
    ph: m.g_tph * e[0] + m.g_phph * e[3],
  };
}

/**
 * Tetrad direction n̂ (inward, n1 < 0) whose traced photon carries exactly
 * L_z/E = xi and p_θ/E = pThetaOverE. The constraints are linear in n̂
 * (p_μ(n̂) = g·(e0 + nⁱeᵢ) is affine), leaving a line of solutions; the unit
 * norm picks two points and the inward root is the shadow-relevant one.
 * Returns null when no real inward solution exists.
 */
export function directionForConserved(
  m: MetricBL,
  tetrad: Tetrad,
  xi: number,
  pThetaOverE: number,
): [number, number, number] | null {
  const P = tetrad.map((e) => lower(m, e));
  // E = −p_t = −(P0.t + Σ nⁱ Pi.t); constraint rows: L − ξE = 0, p_θ − (p_θ/E)·E = 0.
  const rowA = P.map((p) => p.ph + xi * p.t); // coefficients of (1, n1, n2, n3)
  const rowB = P.map((p) => p.th + pThetaOverE * p.t);

  // Solve rowA·(1,n) = 0, rowB·(1,n) = 0 for n on the affine line n_p + t·d.
  const a1 = rowA[1]!, a2 = rowA[2]!, a3 = rowA[3]!, a0 = rowA[0]!;
  const b1 = rowB[1]!, b2 = rowB[2]!, b3 = rowB[3]!, b0 = rowB[0]!;
  // Null direction d = rA × rB of the two constraint normals.
  const d: [number, number, number] = [
    a2 * b3 - a3 * b2,
    a3 * b1 - a1 * b3,
    a1 * b2 - a2 * b1,
  ];
  const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  if (!(dd > 0)) return null;
  // Particular solution: least-norm solve of the 2×3 system A n = −(a0, b0).
  const g11 = a1 * a1 + a2 * a2 + a3 * a3;
  const g12 = a1 * b1 + a2 * b2 + a3 * b3;
  const g22 = b1 * b1 + b2 * b2 + b3 * b3;
  const det = g11 * g22 - g12 * g12;
  if (!(Math.abs(det) > 0)) return null;
  const la = (-a0 * g22 - -b0 * g12) / det;
  const lb = (g11 * -b0 - g12 * -a0) / det;
  const np: [number, number, number] = [
    la * a1 + lb * b1,
    la * a2 + lb * b2,
    la * a3 + lb * b3,
  ];
  // |n_p + t d| = 1 → quadratic in t; pick the inward (n1 < 0) root.
  const npd = np[0] * d[0] + np[1] * d[1] + np[2] * d[2];
  const npnp = np[0] * np[0] + np[1] * np[1] + np[2] * np[2];
  const disc = npd * npd - dd * (npnp - 1);
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  for (const t of [(-npd - sq) / dd, (-npd + sq) / dd]) {
    const n: [number, number, number] = [np[0] + t * d[0], np[1] + t * d[1], np[2] + t * d[2]];
    if (n[0] < 0) return n;
  }
  return null;
}

/** Screen-pixel position of a tetrad direction under the renderer's pinhole
 * model (mirrors src/shaders/trace/init.wgsl exactly). */
export function directionToPixel(
  n: [number, number, number],
  framing: CameraFraming,
): { x: number; y: number } {
  const tanHalfFovY = Math.tan(framing.fovY / 2);
  const tanHalfFovX = (tanHalfFovY * framing.width) / framing.height;
  const ndcX = n[2] / (-n[0] * tanHalfFovX);
  const ndcY = n[1] / (n[0] * tanHalfFovY);
  return {
    x: ((ndcX + 1) / 2) * framing.width - 0.5,
    y: ((1 - ndcY) / 2) * framing.height - 0.5,
  };
}

/**
 * The analytic shadow contour in screen-pixel coordinates for the ZAMO camera
 * at the given framing: Bardeen family (ξ, η) → exact tetrad direction (both
 * p_θ signs) → pinhole pixel. Points outside the field of view are dropped.
 */
export function analyticShadowContourPixels(
  a: number,
  framing: CameraFraming,
  samples = 512,
): Array<{ x: number; y: number }> {
  const m = createMetric();
  evalMetric(a, framing.r, framing.theta, m);
  const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, a, framing.r));
  const points: Array<{ x: number; y: number }> = [];
  for (const { xi, eta } of bardeenContour(a, samples)) {
    for (const sign of eta > 0 ? [1, -1] : [1]) {
      // p_θ/E = ±√η exactly on the equatorial plane (Q = p_θ² there); off the
      // equatorial plane Q picks up the cos²θ terms, handled via η directly:
      // p_θ²/E² = η + cos²θ(a² − ξ²/sin²θ) at the camera position.
      const c = Math.cos(framing.theta);
      const s = Math.sin(framing.theta);
      const pth2 = eta + c * c * (a * a - (xi * xi) / (s * s));
      if (pth2 < 0) continue; // this family point is not visible at this inclination
      const n = directionForConserved(m, tetrad, xi, sign * Math.sqrt(pth2));
      if (!n) continue;
      const px = directionToPixel(n, framing);
      if (px.x < -0.5 || px.x > framing.width - 0.5 || px.y < -0.5 || px.y > framing.height - 0.5) continue;
      points.push(px);
    }
  }
  return points;
}
