// Pass A — ray init (ARCHITECTURE.md §4): pixel -> pinhole direction in the
// observer tetrad -> k = e0 + n^i e_i -> p_mu = g_{mu nu} k^nu (PHYSICS.md §5).
// The tetrad and camera metric arrive precomputed in float64 via uniforms.

#include "../common/uniforms.wgsl"
#include "../common/ray-state.wgsl"

override WORKGROUP_SIZE: u32 = 8u;

// Float32 boundary (PHYSICS.md §10): a ray whose polar turning layer
// s_min = |p_phi|/P sits below this cannot be resolved by f32 theta near
// theta = pi (ULP(pi) = 2.4e-7; measured misclassification wall ~ 8e-6), and
// it differs from the exact L = 0 polar ray by less than f32 roundoff in
// every observable. Such rays are quantized onto the axis-regular L = 0
// branch at init (p_phi is exactly conserved, so init-time suffices).
const AXIAL_SNAP_SMIN: f32 = 4e-5;

// Plane-launch quantization (twin of PLANE_LAUNCH_EPS in
// src/physics/integrator.ts — change both together): a camera within
// roundoff of the equatorial plane reads cos(theta0) at ULP scale with an
// arbitrary sign, so rays would self-intersect the disk at their own launch
// point (sigma = 0) whenever the camera radius lies inside the disk band.
// Launch theta is quantized onto the ray's departure side (dtheta ∝ p_theta).
const PLANE_LAUNCH_EPS: f32 = 1e-6;
const HALF_PI: f32 = 1.5707963267948966;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn cs_init(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.resolution.x || gid.y >= u.resolution.y) {
    return;
  }
  let i = gid.y * u.resolution.x + gid.x;

  // NDC with +y up; screen up looks toward decreasing theta (-e2).
  // u.sample.xy: sub-pixel jitter for temporal accumulation (M6); zero for
  // the base sample, so the first converged image is exactly the pixel-center
  // trace that every V-O/V-I comparison uses.
  // u.tile (M7): this grid may be a tile of a larger image (tiled export);
  // NDC is computed in full-image pixel coordinates. Interactive rendering
  // uses tile = (0, 0, resolution), reproducing the untiled mapping
  // bit-exactly (adding 0.0 and dividing by the same f32 value).
  let ndcX = (2.0 * (u.tile.x + f32(gid.x) + 0.5 + u.sample.x)) / u.tile.z - 1.0;
  let ndcY = 1.0 - (2.0 * (u.tile.y + f32(gid.y) + 0.5 + u.sample.y)) / u.tile.w;
  let n = normalize(vec3f(-1.0, -ndcY * u.camMetricB.z, ndcX * u.camMetricB.y));

  // k^mu in BL coordinates from the tetrad legs.
  let k = u.e0 + n.x * u.e1 + n.y * u.e2 + n.z * u.e3;

  // Lower the index with the covariant camera metric.
  let g_tt = u.camMetricA.x;
  let g_tph = u.camMetricA.y;
  let g_rr = u.camMetricA.z;
  let g_thth = u.camMetricA.w;
  let g_phph = u.camMetricB.x;

  var y: array<f32, 8>;
  y[IT] = 0.0;
  y[IR] = u.camPos.x;
  y[ITH] = u.camPos.y;
  y[IPH] = u.camPos.z;
  y[IPT] = g_tt * k.x + g_tph * k.w;
  y[IPR] = g_rr * k.y;
  y[IPTH] = g_thth * k.z;
  y[IPPH] = g_tph * k.x + g_phph * k.w;

  // Polar-plane quantization (see AXIAL_SNAP_SMIN above).
  let sCam = sin(y[ITH]);
  let lOverS = y[IPPH] / sCam;
  let P = sqrt(y[IPTH] * y[IPTH] + lOverS * lOverS);
  if (abs(y[IPPH]) < AXIAL_SNAP_SMIN * P) {
    y[IPPH] = 0.0;
  }

  // Disk plane-launch quantization (see PLANE_LAUNCH_EPS above); disabled
  // disk (rOuter = 0) has no plane to hit.
  if (u.disk.y > 0.0 && y[IPTH] != 0.0 && abs(y[ITH] - HALF_PI) < PLANE_LAUNCH_EPS) {
    y[ITH] = HALF_PI + sign(y[IPTH]) * PLANE_LAUNCH_EPS;
  }
  storeRayState(i, y);

  state[P_H * u.rayCount + i] = 0.0; // sentinel: march computes the initial step
  state[P_ERR_OLD * u.rayCount + i] = 1e-4;
  state[P_STEPS * u.rayCount + i] = 0.0;
  status[i] = STATUS_MARCHING;
}
