// Disk kinematics and redshift — float32 ports of src/physics/orbits.ts and
// the exact redshift contraction (PHYSICS.md §6, §8). Valid for prograde
// circular equatorial emitters at r >= r_isco (the disk's own domain), where
// the stability radicand r^{3/2} - 3 r^{1/2} + 2a is strictly positive.

// Omega(r) = 1 / (r^{3/2} + a)
fn diskAngularVelocity(a: f32, r: f32) -> f32 {
  return 1.0 / (pow(r, 1.5) + a);
}

// u^t = (r^{3/2} + a) / (r^{3/4} * sqrt(r^{3/2} - 3 r^{1/2} + 2a))
fn diskCircularUt(a: f32, r: f32) -> f32 {
  let sr = sqrt(r);
  let radicand = r * sr - 3.0 * sr + 2.0 * a;
  return (r * sr + a) / (pow(r, 0.75) * sqrt(radicand));
}

// g = nu_obs/nu_em = 1 / (u^t (E - Omega L)). Exact for any tetrad camera:
// photonFromTetrad normalizes (-k.u)_obs = 1, and
// (-k.u)_em = u^t (E - Omega L) for the circular emitter (PHYSICS.md §8;
// identity verified in tests/unit/physics/redshift.test.ts).
fn diskRedshift(a: f32, r: f32, E: f32, L: f32) -> f32 {
  return 1.0 / (diskCircularUt(a, r) * (E - diskAngularVelocity(a, r) * L));
}
