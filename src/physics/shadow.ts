/**
 * PHYSICS.md §3 — Bardeen's analytic Kerr shadow contour [R1]. Validation
 * benchmark only (V-A, V-I1); never part of the render path.
 *
 * Spherical photon orbits at BL radius r carry the conserved ratios
 *
 *   ξ(r) = L_z/E = (r²(3 − r) − a²(r + 1)) / (a(r − 1))
 *   η(r) = Q/E²  = r³(4a² − r(r − 3)²) / (a²(r − 1)²)
 *
 * obtained from R(r) = R'(r) = 0 for the radial potential
 * R = [E(r² + a²) − a L_z]² − Δ[(L_z − aE)² + Q]. The shadow boundary seen by
 * any observer outside the photon shell is the image of this one-parameter
 * family (conserved quantities are observer-independent; only the mapping to
 * the local sky depends on the observer — src/validation/shadowContour.ts).
 * η vanishes exactly at the equatorial photon orbits, which bound the family.
 * At a = 0 the family degenerates to r = 3 with ξ² + η = 27 (b_c = 3√3, V-A3).
 */
import { photonOrbitRadius } from './orbits';

const B_CRIT_SCHWARZSCHILD = 3 * Math.sqrt(3);

export interface ShadowContourPoint {
  /** L_z/E of the (arriving) boundary photon. */
  xi: number;
  /** Q/E² of the boundary photon (≥ 0 over the equatorial-observer contour). */
  eta: number;
}

/** ξ, η of the spherical photon orbit at BL radius r (a > 0). */
export function sphericalPhotonConserved(a: number, r: number): ShadowContourPoint {
  if (!(a > 0)) {
    throw new RangeError('sphericalPhotonConserved requires a > 0 (a = 0 degenerates to r = 3).');
  }
  const a2 = a * a;
  const rm1 = r - 1;
  const xi = (r * r * (3 - r) - a2 * (r + 1)) / (a * rm1);
  const eta = (r * r * r * (4 * a2 - r * (r - 3) * (r - 3))) / (a2 * rm1 * rm1);
  return { xi, eta };
}

/**
 * Critical equatorial impact parameter b_c = ξ at the equatorial photon orbit:
 * the capture/escape boundary for equatorial photons (positive for prograde,
 * negative for retrograde; ±3√3 at a = 0).
 */
export function criticalImpactParameter(a: number, prograde: boolean): number {
  if (a === 0) return prograde ? B_CRIT_SCHWARZSCHILD : -B_CRIT_SCHWARZSCHILD;
  return sphericalPhotonConserved(a, photonOrbitRadius(a, prograde)).xi;
}

/**
 * Sample the shadow-boundary family (ξ(r), η(r)) over the photon shell
 * [r_ph^prograde, r_ph^retrograde], cosine-spaced so the equatorial endpoints
 * (η = 0, where the contour closes for an equatorial observer) are resolved.
 * At a = 0: the circle ξ = b_c cos ψ, η = b_c² sin²ψ.
 */
export function bardeenContour(a: number, samples: number): ShadowContourPoint[] {
  const points: ShadowContourPoint[] = [];
  if (a === 0) {
    for (let i = 0; i < samples; i++) {
      const psi = (Math.PI * i) / (samples - 1);
      const xi = B_CRIT_SCHWARZSCHILD * Math.cos(psi);
      points.push({ xi, eta: B_CRIT_SCHWARZSCHILD * B_CRIT_SCHWARZSCHILD - xi * xi });
    }
    return points;
  }
  const rPro = photonOrbitRadius(a, true);
  const rRetro = photonOrbitRadius(a, false);
  for (let i = 0; i < samples; i++) {
    const t = 0.5 * (1 - Math.cos((Math.PI * i) / (samples - 1)));
    const r = rPro + t * (rRetro - rPro);
    const { xi, eta } = sphericalPhotonConserved(a, r);
    // Endpoint rounding can make η an epsilon negative; it is exactly 0 there.
    points.push({ xi, eta: Math.max(eta, 0) });
  }
  return points;
}
