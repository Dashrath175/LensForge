// Shared trace uniforms (bound at @group(0) @binding(0) in every trace pass).
// Camera-frame quantities (tetrad, covariant metric at the camera) are
// computed once per camera change in float64 on the CPU (src/physics) and
// uploaded — per-pixel GPU work never rebuilds them.

struct TraceUniforms {
  // Observer tetrad legs, contravariant BL components (t, r, theta, phi).
  e0: vec4f,
  e1: vec4f,
  e2: vec4f,
  e3: vec4f,
  // Covariant metric at the camera: (g_tt, g_tph, g_rr, g_thth) and (g_phph, -, -, -).
  camMetricA: vec4f,
  camMetricB: vec4f, // x: g_phph, y: tanHalfFovX, z: tanHalfFovY, w: spin a
  camPos: vec4f,     // x: r0, y: theta0, z: phi0, w: marker inner cos
  markerDir: vec4f,  // xyz: antipodal sky marker direction, w: marker outer cos
  resolution: vec2u,
  rayCount: u32,
  stepsPerDispatch: u32,
  rtol: f32,
  atol: f32,
  rCapture: f32,
  rEscape: f32,
}

@group(0) @binding(0) var<uniform> u: TraceUniforms;
