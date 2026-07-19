// M7 production sky (ARCHITECTURE.md D9): HDR cubemap background + analytic
// bright-star splats scaled by |det J|⁻¹.
//
// The cubemap (tools/generateSkyAssets.ts) carries only *extended* radiance —
// diffuse Milky Way, zodiacal glow, and the unresolved star field with every
// bright catalog star photometrically subtracted. Sampling it by the ray's
// asymptotic direction is exact under lensing: gravitational lensing
// conserves surface brightness, so no magnification factor may appear here.
//
// Point sources are different: a star's image is a delta function whose
// *flux* scales by the lensing magnification μ = ω_lin²/|det J|, where J is
// the Jacobian of the screen→sky map (radians per pixel) and ω_lin the
// unlensed angular pixel pitch. Splatting a star with a screen-space Gaussian
// PSF G (normalized to unit integral over pixel area) gives the radiance
//
//   L(p) = Σ_i  E_i · G(J⁻¹ Δs_i) / |det J|,
//
// where E_i is the star's unlensed irradiance (units: stored-radiance · sr,
// measured from the same panorama the cubemap comes from — one unit system by
// construction) and Δs_i the star's angular offset from the pixel's sky
// direction in an orthonormal tangent basis. Summed over pixels this delivers
// exactly E_i · μ: the |det J|⁻¹ magnification of D9. J is finite-differenced
// from neighboring rays' escape directions (the documented approach; noisy at
// the shadow edge, so |det J| is clamped — see DET_CLAMP_MU below).
//
// When a pixel's sky footprint grows past the star-grid cell size (strong
// demagnification near the shadow: one pixel sees steradians of sky), the
// splat sum smoothly hands over to a low-resolution aggregate cubemap holding
// the same catalog stars' radiance — the surface-brightness-correct limit of
// an unresolved star field. The blend conserves energy: splats fade with
// (1 − t) exactly as the aggregate fades in with t.

// Star-grid layout — mirrored in tools/generateSkyAssets.ts (change both).
const SKY_GRID_NTH: u32 = 32u;
const SKY_GRID_NPH: u32 = 64u;
const STAR_STRIDE: u32 = 8u;
const PI_SKY: f32 = 3.141592653589793;
const TWO_PI_SKY: f32 = 6.283185307179586;
// Magnification clamp (IMPLEMENTATION.md M7 risk): |det J| is floored at
// ω_lin²/DET_CLAMP_MU, capping point-source magnification at DET_CLAMP_MU.
const DET_CLAMP_MU: f32 = 1e4;
// Splat→aggregate handover band, radians of sky-footprint radius. The upper
// edge stays below the theta cell size (π/32 ≈ 0.098), bounding the cell scan.
const SPLAT_RHO0: f32 = 0.045;
const SPLAT_RHO1: f32 = 0.09;

// Traced-photon → arriving-photon parity (M7, VALIDATION.md §6): the render
// traces photons *leaving* the camera; the physical image is of photons
// *arriving*. Kerr is invariant under J: (t, φ) → (−t, −φ), and J∘(affine
// reversal) maps each arriving photon exactly onto a traced photon with the
// φ-mirrored screen coordinate — same r/θ path, same E, L, same redshift
// contraction (isometry), φ-mirrored asymptotics. So the physical image is
// the traced image mirrored in screen x (present.wgsl) with the sky sampled
// at the φ-mirrored escape direction (this function). Exact, no
// approximation; the disk needs no change because its emission is
// axisymmetric and the contraction −p·u is J-invariant.
fn arrivalDir(dTraced: vec3f) -> vec3f {
  return vec3f(dTraced.x, -dTraced.y, dTraced.z);
}

@group(0) @binding(11) var skyCube: texture_cube<f32>;
@group(0) @binding(12) var skySampler: sampler;
@group(0) @binding(13) var starAggCube: texture_cube<f32>;
// Flat star records [dirx,diry,dirz, fluxR,fluxG,fluxB, 0,0], sorted by cell.
@group(0) @binding(14) var<storage, read> skyStars: array<f32>;
// CSR offsets into skyStars per (theta, phi) cell; length NTH*NPH + 1.
@group(0) @binding(15) var<storage, read> skyCellStart: array<u32>;

/** Extended-source background: direct cubemap lookup (exact under lensing). */
fn skyBackground(dir: vec3f) -> vec3f {
  return textureSampleLevel(skyCube, skySampler, dir, 0.0).rgb * u.sky.x;
}

/** Aggregated catalog-star radiance (demagnified/unresolved regime). */
fn skyStarAggregate(dir: vec3f) -> vec3f {
  return textureSampleLevel(starAggCube, skySampler, dir, 0.0).rgb * u.sky.x;
}

struct SkyJacobian {
  // Columns: d(sky dir)/d(screen px) along +x and +y, radians (tangent plane).
  jx: vec2f,
  jy: vec2f,
  valid: bool,
}

/**
 * Bright-star splat sum for one pixel. dir: unit sky direction; (jxv, jyv):
 * ∂dir/∂px along screen x/y (3-vectors from neighbor finite differences).
 */
fn starSplats(dir: vec3f, jxv: vec3f, jyv: vec3f) -> vec3f {
  // Orthonormal tangent basis at dir.
  let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(dir.z) > 0.9);
  let t1 = normalize(cross(up, dir));
  let t2 = cross(dir, t1);
  let a = dot(jxv, t1);
  let c = dot(jxv, t2);
  let b = dot(jyv, t1);
  let d = dot(jyv, t2);

  // Unlensed pixel pitch of the FULL image (tile.w = full height in px).
  let omegaLin = 2.0 * u.camMetricB.z / u.tile.w;
  var det = a * d - b * c;
  let detFloor = omegaLin * omegaLin / DET_CLAMP_MU;
  det = sign(det) * max(abs(det), detFloor);
  if (det == 0.0) {
    det = detFloor;
  }

  let sigma = u.sky.y;
  // Sky-footprint radius of the splat kernel (3σ in px through J).
  let rho = 3.0 * sigma * sqrt(dot(jxv, jxv) + dot(jyv, jyv));
  let t = smoothstep(SPLAT_RHO0, SPLAT_RHO1, rho);
  var radiance = t * skyStarAggregate(dir);
  if (t >= 1.0) {
    return radiance;
  }

  // Inverse Jacobian (sky → screen px).
  let invDet = 1.0 / det;
  let gaussNorm = (1.0 - t) / (TWO_PI_SKY * sigma * sigma * abs(det));

  // Cell scan over the spherical cap (dir, rho) in the (theta, phi) grid.
  let theta = acos(clamp(dir.z, -1.0, 1.0));
  var phi = atan2(dir.y, dir.x);
  if (phi < 0.0) {
    phi += TWO_PI_SKY;
  }
  let cellTh = PI_SKY / f32(SKY_GRID_NTH);
  let cellPh = TWO_PI_SKY / f32(SKY_GRID_NPH);
  let itLo = u32(clamp((theta - rho) / cellTh, 0.0, f32(SKY_GRID_NTH - 1u)));
  let itHi = u32(clamp((theta + rho) / cellTh, 0.0, f32(SKY_GRID_NTH - 1u)));
  var splat = vec3f(0.0);
  for (var it = itLo; it <= itHi; it++) {
    // Azimuthal half-span widened by 1/sin(theta); sin is concave on [0, π],
    // so its minimum over the row×cap intersection is at an endpoint.
    let rowLo = max(f32(it) * cellTh, theta - rho);
    let rowHi = min(f32(it + 1u) * cellTh, theta + rho);
    let sinMin = min(sin(rowLo), sin(rowHi));
    var nPhiCells = SKY_GRID_NPH;
    var ipLo = 0u;
    if (rho < sinMin * PI_SKY) {
      let dPhi = rho / max(sinMin, 1e-4);
      let span = u32(ceil(dPhi / cellPh));
      nPhiCells = min(SKY_GRID_NPH, 2u * span + 1u);
      let ipC = u32(clamp(phi / cellPh, 0.0, f32(SKY_GRID_NPH - 1u)));
      ipLo = (ipC + SKY_GRID_NPH - span) % SKY_GRID_NPH;
    }
    for (var k = 0u; k < nPhiCells; k++) {
      let ip = (ipLo + k) % SKY_GRID_NPH;
      let cell = it * SKY_GRID_NPH + ip;
      let s0 = skyCellStart[cell];
      let s1 = skyCellStart[cell + 1u];
      for (var s = s0; s < s1; s++) {
        let base = s * STAR_STRIDE;
        let sdir = vec3f(skyStars[base], skyStars[base + 1u], skyStars[base + 2u]);
        // Tangent-plane angular offset (small-angle; rho ≤ 0.09 rad here).
        let delta = sdir - dir;
        let s1c = dot(delta, t1);
        let s2c = dot(delta, t2);
        // Screen-space offset q = J⁻¹ Δs, pixels.
        let qx = (d * s1c - b * s2c) * invDet;
        let qy = (-c * s1c + a * s2c) * invDet;
        let r2 = qx * qx + qy * qy;
        if (r2 < 9.0 * sigma * sigma * 2.25) { // 4.5σ screen cutoff
          let flux = vec3f(skyStars[base + 3u], skyStars[base + 4u], skyStars[base + 5u]);
          splat += flux * exp(-0.5 * r2 / (sigma * sigma));
        }
      }
    }
  }
  return radiance + splat * gaussNorm * u.sky.x;
}
