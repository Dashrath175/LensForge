/**
 * PHYSICS.md §3, §6 — circular equatorial orbits and characteristic radii
 * (prograde, a ∈ [0, 1)), from Bardeen, Press & Teukolsky [R10]. G = c = M = 1.
 */

/** Orbital angular velocity Ω(r) = 1/(r^{3/2} + a). */
export function angularVelocity(a: number, r: number): number {
  return 1 / (Math.pow(r, 1.5) + a);
}

/** √ of the circular-orbit stability radicand r^{3/2} − 3r^{1/2} + 2a (must be > 0 for orbits to exist). */
function orbitRadicandRoot(a: number, r: number): number {
  const sr = Math.sqrt(r);
  const radicand = r * sr - 3 * sr + 2 * a;
  if (radicand <= 0) {
    throw new RangeError(`No timelike circular orbit at r=${r}, a=${a} (inside photon orbit).`);
  }
  return Math.sqrt(radicand);
}

/** u^t of a circular orbit: (r^{3/2} + a) / (r^{3/4} √(r^{3/2} − 3r^{1/2} + 2a)). */
export function circularUt(a: number, r: number): number {
  return (Math.pow(r, 1.5) + a) / (Math.pow(r, 0.75) * orbitRadicandRoot(a, r));
}

/** Specific orbital energy E†(r) = (r^{3/2} − 2r^{1/2} + a) / (r^{3/4} √(r^{3/2} − 3r^{1/2} + 2a)). */
export function specificEnergy(a: number, r: number): number {
  const sr = Math.sqrt(r);
  return (r * sr - 2 * sr + a) / (Math.pow(r, 0.75) * orbitRadicandRoot(a, r));
}

/** Specific angular momentum L†(r) = (r² − 2a r^{1/2} + a²) / (r^{3/4} √(r^{3/2} − 3r^{1/2} + 2a)). */
export function specificAngularMomentum(a: number, r: number): number {
  return (r * r - 2 * a * Math.sqrt(r) + a * a) / (Math.pow(r, 0.75) * orbitRadicandRoot(a, r));
}

/** Contravariant 4-velocity u^μ = u^t (1, 0, 0, Ω) of a circular orbit. */
export function circularFourVelocity(a: number, r: number): [number, number, number, number] {
  const ut = circularUt(a, r);
  return [ut, 0, 0, ut * angularVelocity(a, r)];
}

/** ISCO radius, closed form [R10] (prograde branch). */
export function iscoRadius(a: number): number {
  const a2 = a * a;
  const z1 = 1 + Math.cbrt(1 - a2) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a2 + z1 * z1);
  return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

/** Equatorial circular photon orbit radius, closed form [R10]. */
export function photonOrbitRadius(a: number, prograde = true): number {
  const arg = prograde ? -a : a;
  return 2 * (1 + Math.cos((2 / 3) * Math.acos(arg)));
}
