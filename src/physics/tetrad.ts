/**
 * PHYSICS.md §5 — observer 4-velocities and local orthonormal tetrads.
 *
 * Tetrads are built by metric Gram-Schmidt over {u, ∂_r, ∂_θ, ∂_φ}, which
 * guarantees g(e_a, e_b) = η_ab for every observer type by construction
 * (verified in V-A1). Relativistic aberration lives entirely in the choice
 * of u — never in separate code (MASTER_PLAN.md §5).
 */
import { ergosphereRadius, horizonRadius, innerProduct, type MetricBL } from './metric';
import { angularVelocity, circularFourVelocity } from './orbits';

export type FourVector = [number, number, number, number];

/** Tetrad legs e_0..e_3 as contravariant BL components; e_0 is the 4-velocity. */
export type Tetrad = [FourVector, FourVector, FourVector, FourVector];

/** Static observer u = ∂_t/√(−g_tt); exists only outside the ergosphere. */
export function staticFourVelocity(m: MetricBL, a: number, r: number, theta: number): FourVector {
  if (r <= ergosphereRadius(a, theta)) {
    throw new RangeError(`Static observer does not exist at r=${r}, θ=${theta} (inside ergosphere).`);
  }
  return [1 / Math.sqrt(-m.g_tt), 0, 0, 0];
}

/** ZAMO u ∝ ∂_t + ω∂_φ with ω = −g_tφ/g_φφ; the default observer (valid outside the horizon). */
export function zamoFourVelocity(m: MetricBL, a: number, r: number): FourVector {
  if (r <= horizonRadius(a)) {
    throw new RangeError(`ZAMO does not exist at r=${r} (inside horizon).`);
  }
  const omega = -m.g_tph / m.g_phph;
  const norm2 = -(m.g_tt + 2 * omega * m.g_tph + omega * omega * m.g_phph);
  const ut = 1 / Math.sqrt(norm2);
  return [ut, 0, 0, ut * omega];
}

/** Circular-orbit observer (equatorial, prograde); PHYSICS.md §6. */
export function circularOrbitFourVelocity(a: number, r: number): FourVector {
  return circularFourVelocity(a, r);
}

export { angularVelocity };

/**
 * Orthonormal tetrad from a normalized 4-velocity u by Gram-Schmidt of the
 * coordinate basis vectors ∂_r, ∂_θ, ∂_φ against {u, ...} using the metric
 * inner product. Component of v along a unit leg e_a is η_aa⟨v, e_a⟩.
 */
export function orthonormalTetrad(m: MetricBL, u: FourVector): Tetrad {
  const legs: FourVector[] = [[...u]];
  const seeds: FourVector[] = [
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  for (const seed of seeds) {
    const v: FourVector = [...seed];
    for (let aIdx = 0; aIdx < legs.length; aIdx++) {
      const eta = aIdx === 0 ? -1 : 1;
      const coeff = eta * innerProduct(m, v, legs[aIdx]!);
      for (let mu = 0; mu < 4; mu++) v[mu]! -= coeff * legs[aIdx]![mu]!;
    }
    const norm2 = innerProduct(m, v, v);
    if (!(norm2 > 0)) {
      throw new RangeError('Tetrad construction degenerate (spacelike leg has non-positive norm).');
    }
    const inv = 1 / Math.sqrt(norm2);
    for (let mu = 0; mu < 4; mu++) v[mu]! *= inv;
    legs.push(v);
  }
  return legs as Tetrad;
}
