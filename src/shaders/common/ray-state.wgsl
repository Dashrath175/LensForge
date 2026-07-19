// Persistent ray state, SoA plane layout (ARCHITECTURE.md §5): plane p of ray
// i lives at state[p * u.rayCount + i]. Layout constants are mirrored in
// src/renderer/tracePass.ts — change both together.
//
// State vector order matches src/physics/hamiltonian.ts:
//   y = [t, r, theta, phi, p_t, p_r, p_theta, p_phi]

// Plane indices.
const P_Y0: u32 = 0u;      // ..P_Y7 = 7 (geodesic state)
const P_H: u32 = 8u;       // current step size (0 = uninitialized sentinel)
const P_ERR_OLD: u32 = 9u; // previous error norm (PI controller memory)
const P_STEPS: u32 = 10u;  // accepted-step count
const STATE_PLANES: u32 = 11u;
// Per-status result payload: ESCAPED -> asymptotic direction xyz;
// HIT_DISK -> (r_hit, phi_hit, g). Never both for one ray.
const RESULT_PLANES: u32 = 3u;

// Status codes (mirrored in tracePass.ts).
const STATUS_MARCHING: u32 = 0u;
const STATUS_CAPTURED: u32 = 1u;
const STATUS_ESCAPED: u32 = 2u;
const STATUS_MAX_STEPS: u32 = 3u;
const STATUS_HIT_DISK: u32 = 4u;

// State component indices (PHYSICS.md §4).
const IT = 0;
const IR = 1;
const ITH = 2;
const IPH = 3;
const IPT = 4;
const IPR = 5;
const IPTH = 6;
const IPPH = 7;

@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> status: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

fn loadRayState(i: u32) -> array<f32, 8> {
  var y: array<f32, 8>;
  for (var p = 0u; p < 8u; p++) {
    y[p] = state[p * u.rayCount + i];
  }
  return y;
}

fn storeRayState(i: u32, y: array<f32, 8>) {
  for (var p = 0u; p < 8u; p++) {
    state[p * u.rayCount + i] = y[p];
  }
}
