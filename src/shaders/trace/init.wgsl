// Pass A — ray init (ARCHITECTURE.md §4): pixel -> pinhole direction in the
// observer tetrad -> k = e0 + n^i e_i -> p_mu = g_{mu nu} k^nu (PHYSICS.md §5).
// The tetrad and camera metric arrive precomputed in float64 via uniforms.

#include "../common/uniforms.wgsl"
#include "../common/ray-state.wgsl"

override WORKGROUP_SIZE: u32 = 8u;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn cs_init(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.resolution.x || gid.y >= u.resolution.y) {
    return;
  }
  let i = gid.y * u.resolution.x + gid.x;

  // NDC with +y up; screen up looks toward decreasing theta (-e2).
  let ndcX = (2.0 * (f32(gid.x) + 0.5)) / f32(u.resolution.x) - 1.0;
  let ndcY = 1.0 - (2.0 * (f32(gid.y) + 0.5)) / f32(u.resolution.y);
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
  storeRayState(i, y);

  state[P_H * u.rayCount + i] = 0.0; // sentinel: march computes the initial step
  state[P_ERR_OLD * u.rayCount + i] = 1e-4;
  state[P_STEPS * u.rayCount + i] = 0.0;
  status[i] = STATUS_MARCHING;
}
