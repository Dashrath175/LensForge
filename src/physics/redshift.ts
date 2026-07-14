/**
 * PHYSICS.md §8 — redshift factor g = ν_obs/ν_em.
 *
 * For a distant static observer, (−k·u)_obs = E, so g = E / (−k_μ u^μ_em).
 * The contraction is covariant-momentum × contravariant-velocity: no metric
 * required at the emitter.
 */
import { IPH, IPR, IPT, IPTH, IPPH } from './hamiltonian';
import { angularVelocity, circularUt } from './orbits';
import type { FourVector } from './tetrad';

/** g to infinity for an emitter with 4-velocity u at the photon's emission state. */
export function redshiftToInfinity(photonState: Float64Array, uEmitter: FourVector): number {
  const E = -photonState[IPT]!;
  const kDotU =
    photonState[IPT]! * uEmitter[0]! +
    photonState[IPR]! * uEmitter[1]! +
    photonState[IPTH]! * uEmitter[2]! +
    photonState[IPPH]! * uEmitter[3]!;
  return E / -kDotU;
}

/**
 * Compact form for circular disk emitters and a distant static observer
 * (Cunningham [R6]): g = 1 / [u^t (1 − Ω λ)] with λ = L_z/E.
 */
export function diskRedshiftCompact(a: number, r: number, lambdaImpact: number): number {
  return 1 / (circularUt(a, r) * (1 - angularVelocity(a, r) * lambdaImpact));
}
