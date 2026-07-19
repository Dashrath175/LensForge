// Pass C — shade (ARCHITECTURE.md §4): terminated rays -> HDR radiance.
//
// M7 sky (D9): ESCAPED rays sample the production sky — star-subtracted HDR
// cubemap by asymptotic direction (surface brightness conservation makes the
// direct lookup exact, PHYSICS.md §8) plus analytic bright-star splats scaled
// by |det J|⁻¹, with J finite-differenced from the neighboring rays' escape
// directions (shaders/sky/sky.wgsl). Captured rays are black (the shadow).
//
// M4 disk shading (PHYSICS.md §7-§8): HIT_DISK rays evaluate the blackbody
// shortcut — a blackbody stays a blackbody with T_obs = g * T_eff, so the
// Planck->XYZ LUT sampled at T_obs already contains gravitational redshift,
// Doppler shift, AND g^3 beaming exactly; no separate factors exist by
// construction. T_eff comes from the Novikov-Thorne T-hat LUT (sliced to the
// current spin on the CPU, src/renderer/diskLuts.ts) times the physical
// temperature scale in disk.z. disk.w converts the physical W-unit XYZ into
// the frame's HDR working range (a global linear pre-exposure — part of the
// documented camera model, as is the sky radiance scale u.sky.x).
//
// M7 optional disk inhomogeneity (PHYSICS.md §9.6, off by default): a
// statistical brightness field advected at the exact Ω(r). The local flux is
// modulated by m (mean 1), so T_eff scales by m^{1/4} — the emitter stays a
// blackbody and the blackbody shortcut stays exact. The pattern is evaluated
// at the *emission* coordinate time t_sim − t_hit (the geodesic's integrated
// t at the crossing), so differential light-travel retardation across the
// image is physical; the texture itself is a labeled statistical model.

#include "../common/uniforms.wgsl"
#include "../common/ray-state.wgsl"
#include "../disk/kinematics.wgsl"
#include "../sky/sky.wgsl"

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(5) var hdrOut: texture_storage_2d<rgba16float, write>;
// Coarser level of the same trace at the same camera pose (the interaction
// preview, src/renderer/tracePass.ts): still-MARCHING pixels present its
// radiance instead of black, so the composed image is always complete for the
// current pose. The coarsest level binds a 1x1 black dummy (no coarser level
// exists; its unterminated rays are the near-critical set hugging the shadow).
@group(0) @binding(8) var fallbackTex: texture_2d<f32>;
@group(0) @binding(9) var fallbackSampler: sampler;
// Temporal accumulation state (M6), float32 so refinement never quantizes:
// the running mean of jittered samples. u.sample.z = 1 overwrites (base
// trace and interaction); 1/(n+1) folds in refinement round n. The blended
// value is also written to hdrOut, which stays the presented image.
@group(0) @binding(10) var<storage, read_write> accum: array<vec4f>;
// Novikov-Thorne T-hat(r) at the current spin, on the uniform grid
// u = ln(r/rIsco) in [0, lutMeta.x] with lutMeta.y samples.
@group(0) @binding(6) var<storage, read> ntTeffLut: array<f32>;
// Blackbody XYZ on the uniform log10(T) grid [lutMeta.z, lutMeta.w], flat triplets.
@group(0) @binding(7) var<storage, read> planckXyzLut: array<f32>;

// CIE XYZ (D65) -> linear sRGB [R17].
const XYZ_TO_SRGB = mat3x3f(
  vec3f(3.2406, -0.9689, 0.0557),
  vec3f(-1.5372, 1.8758, -0.2040),
  vec3f(-0.4986, 0.0415, 1.0570),
);

fn sampleNtTHat(r: f32) -> f32 {
  let count = arrayLength(&ntTeffLut);
  let uCoord = log(r / u.disk.x);
  let x = clamp(uCoord / u.lutMeta.x, 0.0, 1.0) * f32(count - 1u);
  let i0 = u32(floor(x));
  let i1 = min(i0 + 1u, count - 1u);
  let f = x - floor(x);
  return mix(ntTeffLut[i0], ntTeffLut[i1], f);
}

fn samplePlanckXyz(tK: f32) -> vec3f {
  let count = arrayLength(&planckXyzLut) / 3u;
  let logT = log(max(tK, 1e-6)) / log(10.0);
  let x = clamp((logT - u.lutMeta.z) / (u.lutMeta.w - u.lutMeta.z), 0.0, 1.0) * f32(count - 1u);
  let i0 = u32(floor(x));
  let i1 = min(i0 + 1u, count - 1u);
  let f = x - floor(x);
  let a0 = vec3f(planckXyzLut[3u * i0], planckXyzLut[3u * i0 + 1u], planckXyzLut[3u * i0 + 2u]);
  let a1 = vec3f(planckXyzLut[3u * i1], planckXyzLut[3u * i1 + 1u], planckXyzLut[3u * i1 + 2u]);
  return mix(a0, a1, f);
}

// ---------------------------------------------------------------------------
// Disk inhomogeneity (M7, statistical model — see header). Value noise on a
// (u = ln r/rIsco, φ) lattice, periodic in φ, advected at the exact Ω(r).
// ---------------------------------------------------------------------------

fn hashCell(x: u32, y: u32) -> f32 {
  var h = x * 374761393u + y * 668265263u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h & 0xffffffu) / 16777216.0;
}

/** Periodic value noise: x unbounded (radial), y wraps at nPhi cells. */
fn valueNoisePeriodic(x: f32, y: f32, nPhi: u32) -> f32 {
  let xi = u32(floor(x));
  let yw = y - floor(y / f32(nPhi)) * f32(nPhi); // y mod nPhi, ≥ 0
  let yi = u32(floor(yw)) % nPhi;
  let yi1 = (yi + 1u) % nPhi;
  let fx = smoothstep(0.0, 1.0, x - floor(x));
  let fy = smoothstep(0.0, 1.0, yw - floor(yw));
  let v00 = hashCell(xi, yi);
  let v10 = hashCell(xi + 1u, yi);
  let v01 = hashCell(xi, yi1);
  let v11 = hashCell(xi + 1u, yi1);
  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

/** Flux modulation m(r, φ, t_em), mean 1, amplitude u.sky.z. */
fn diskInhomogeneity(rHit: f32, phiHit: f32, tEm: f32) -> f32 {
  let a = u.camMetricB.w;
  let uc = log(rHit / u.disk.x);
  // Pattern coordinate co-rotating with the local circular flow (exact Ω(r)):
  // shear between annuli emerges from the radial dependence alone.
  let phiP = phiHit - diskAngularVelocity(a, rHit) * tEm;
  // Three octaves; azimuthal cell counts double so the 2π wrap stays exact.
  let n0 = valueNoisePeriodic(uc * 3.0, (phiP / TWO_PI_SKY) * 16.0, 16u);
  let n1 = valueNoisePeriodic(uc * 6.0 + 37.0, (phiP / TWO_PI_SKY) * 32.0, 32u);
  let n2 = valueNoisePeriodic(uc * 12.0 + 91.0, (phiP / TWO_PI_SKY) * 64.0, 64u);
  let n = (0.5 * n0 + 0.3 * n1 + 0.2 * n2);
  return max(1.0 + u.sky.z * (2.0 * n - 1.0), 0.05);
}

fn diskRadiance(rHit: f32, phiHit: f32, g: f32, tHit: f32) -> vec3f {
  var tEff = sampleNtTHat(rHit) * u.disk.z;
  if (u.sky.z > 0.0) {
    // Emission time = arrival (t_sim) minus the geodesic's coordinate flight
    // time to the crossing — differential retardation across the image.
    // −φ: the traced hit sits at the φ-mirror of the physical emission point
    // (arriving-photon parity, see arrivalDir in sky.wgsl), so the pattern is
    // evaluated at the physical azimuth and advects the physical way.
    tEff *= pow(diskInhomogeneity(rHit, -phiHit, u.sky.w - tHit), 0.25);
  }
  let xyz = samplePlanckXyz(g * tEff) * u.disk.w;
  // Out-of-gamut blackbody chromaticities can go slightly negative in sRGB.
  return max(XYZ_TO_SRGB * xyz, vec3f(0.0));
}

const MAX_STEPS_DEBUG_COLOR = vec3f(0.25, 0.0, 0.25);

/** Escape direction of ray index i (valid only when status == ESCAPED). */
fn escapeDir(i: u32) -> vec3f {
  return vec3f(
    result[0u * u.rayCount + i],
    result[1u * u.rayCount + i],
    result[2u * u.rayCount + i],
  );
}

/**
 * ∂dir/∂px along one screen axis by finite differences over neighboring
 * rays' escape directions: central where both neighbors escaped, one-sided
 * where only one did, invalid where neither (shadow-edge pixels — star
 * splats are skipped there; the clamped-magnification regime anyway).
 */
fn dirDerivative(dir: vec3f, iMinus: u32, mValid: bool, iPlus: u32, pValid: bool) -> vec4f {
  if (mValid && pValid) {
    return vec4f(0.5 * (escapeDir(iPlus) - escapeDir(iMinus)), 1.0);
  }
  if (pValid) {
    return vec4f(escapeDir(iPlus) - dir, 1.0);
  }
  if (mValid) {
    return vec4f(dir - escapeDir(iMinus), 1.0);
  }
  return vec4f(0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn cs_shade(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.resolution.x || gid.y >= u.resolution.y) {
    return;
  }
  let i = gid.y * u.resolution.x + gid.x;

  var color = vec3f(0.0); // CAPTURED (shadow)
  switch (status[i]) {
    case STATUS_ESCAPED: {
      let dirTraced = escapeDir(i);
      // Arriving-photon parity (see arrivalDir in sky.wgsl): the sky is
      // sampled at the φ-mirrored direction; the mirror is linear, so the
      // finite-difference derivatives mirror the same way (|det J| invariant).
      let dir = arrivalDir(dirTraced);
      color = skyBackground(dir);
      // Screen→sky Jacobian from the 4-neighborhood (same-status rays only).
      let xm = i - 1u;
      let xp = i + 1u;
      let ym = i - u.resolution.x;
      let yp = i + u.resolution.x;
      let xmV = gid.x > 0u && status[xm] == STATUS_ESCAPED;
      let xpV = gid.x + 1u < u.resolution.x && status[xp] == STATUS_ESCAPED;
      let ymV = gid.y > 0u && status[ym] == STATUS_ESCAPED;
      let ypV = gid.y + 1u < u.resolution.y && status[yp] == STATUS_ESCAPED;
      let dx = dirDerivative(dirTraced, xm, xmV, xp, xpV);
      let dy = dirDerivative(dirTraced, ym, ymV, yp, ypV);
      if (dx.w > 0.0 && dy.w > 0.0) {
        color += starSplats(dir, arrivalDir(dx.xyz), arrivalDir(dy.xyz));
      }
    }
    case STATUS_HIT_DISK: {
      color = diskRadiance(
        result[0u * u.rayCount + i],
        result[1u * u.rayCount + i],
        result[2u * u.rayCount + i],
        state[0u * u.rayCount + i],
      );
    }
    case STATUS_MAX_STEPS: {
      color = MAX_STEPS_DEBUG_COLOR; // diagnostic, counted by V-O3 in M6
    }
    case STATUS_CAPTURED: {
      // Exactly black (the shadow). An explicit case: falling into the
      // MARCHING default would sample the preview fallback, bleeding ladder
      // radiance into converged shadow-edge pixels (M6 defect found by the
      // M7 tile-vs-monolithic consistency check — the documented contract is
      // CAPTURED → 0 and bit-identical converged output, ARCHITECTURE §4-§5).
    }
    default: {
      // Still MARCHING: same-pose fallback level, never black (see fallbackTex).
      // Both pixel grids sample the same tan-space frustum uniformly, so the
      // bilinear upsample is exactly aligned. The uv is formed in FULL-image
      // space through the tile words so a slab grid (adapter-limit splitting,
      // tracePass.ts createGrid) samples the full-frame fallback at its true
      // position; an untiled grid has tile = (0, 0, resolution), and adding
      // 0.0 / dividing by the same values is bit-identical to the untiled
      // mapping (the init.wgsl tile-NDC argument, M7).
      let uv = (u.tile.xy + vec2f(gid.xy) + vec2f(0.5)) / u.tile.zw;
      color = textureSampleLevel(fallbackTex, fallbackSampler, uv, 0.0).rgb;
    }
  }
  let blended = mix(accum[i].rgb, color, u.sample.z);
  accum[i] = vec4f(blended, 1.0);
  textureStore(hdrOut, gid.xy, vec4f(blended, 1.0));
}
