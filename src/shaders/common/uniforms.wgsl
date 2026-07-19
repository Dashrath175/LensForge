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
  camPos: vec4f,     // x: r0, y: theta0, z: phi0, w: unused
  // Tiled rendering (M7 export; ARCHITECTURE.md §5 — one scheduler code path):
  // xy = pixel offset of this grid inside the full image, zw = full-image
  // pixel dimensions. Interactive rendering always uses (0, 0, resolution),
  // which reproduces the untiled NDC mapping bit-exactly.
  tile: vec4f,
  resolution: vec2u,
  rayCount: u32,
  stepsPerDispatch: u32,
  rtol: f32,
  atol: f32,
  rCapture: f32,
  rEscape: f32,
  // Novikov-Thorne disk (M4). disk.y (rOuter) == 0 disables the disk.
  disk: vec4f,    // x: rIsco (inner edge), y: rOuter, z: display temperature scale K, w: radiance scale (Pass C)
  lutMeta: vec4f, // x: uMax of the nt-teff log-radius grid, y: unused, z: log10 T min, w: log10 T max (planck-xyz grid); sample counts come from arrayLength
  // Temporal accumulation (M6): x, y = sub-pixel ray jitter in pixel units
  // (0 for the base sample — bit-identical to the unjittered trace);
  // z = accumulation weight (1 overwrites, 1/(n+1) blends refinement round n).
  sample: vec4f,
  // M7 sky + disk-inhomogeneity words:
  //   x: sky radiance scale (documented camera-model dial — maps the
  //      normalized panorama units into the HDR working range),
  //   y: bright-star splat PSF sigma, pixels,
  //   z: disk inhomogeneity amplitude (0 = off; labeled statistical model),
  //   w: simulation coordinate time t_sim (GM/c³ units) for advection.
  sky: vec4f,
}

@group(0) @binding(0) var<uniform> u: TraceUniforms;
