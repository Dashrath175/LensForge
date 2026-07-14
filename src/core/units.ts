/**
 * Geometrized-unit conversions (PHYSICS.md §1). Kernel quantities are
 * dimensionless (G = c = M = 1); these functions attach physical scales.
 */
import {
  GRAVITATIONAL_CONSTANT as G,
  SPEED_OF_LIGHT as c,
  STEFAN_BOLTZMANN as sigmaSB,
} from './constants';

/** Gravitational length GM/c^2 in meters, for black hole mass in kg. */
export function gravitationalRadiusMeters(massKg: number): number {
  return (G * massKg) / (c * c);
}

/** Gravitational time GM/c^3 in seconds. */
export function gravitationalTimeSeconds(massKg: number): number {
  return (G * massKg) / (c * c * c);
}

/**
 * Physical disk flux (W m^-2) from code flux F̂ (Page-Thorne per unit Ṁ,
 * PHYSICS.md §7): F_phys = F̂ · Ṁ c^6 / (G M)^2.
 */
export function diskFluxWattsPerM2(fluxCode: number, massKg: number, mdotKgPerS: number): number {
  const gm = G * massKg;
  return (fluxCode * mdotKgPerS * c ** 6) / (gm * gm);
}

/** Effective disk temperature in kelvin: T = (F/σ)^(1/4). Carries the T ∝ Ṁ^(1/4) M^(-1/2) scaling. */
export function diskEffectiveTemperatureK(
  fluxCode: number,
  massKg: number,
  mdotKgPerS: number,
): number {
  return Math.pow(diskFluxWattsPerM2(fluxCode, massKg, mdotKgPerS) / sigmaSB, 0.25);
}
