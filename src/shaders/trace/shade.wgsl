// Pass C — shade (ARCHITECTURE.md §4): terminated rays -> HDR radiance.
// M2 sky is a procedural placeholder (IMPLEMENTATION.md M2): a 15-degree
// celestial graticule plus a bright disc at the camera's antipode, whose
// lensed image is the Einstein ring. Escaped rays sample it through their
// asymptotic direction; captured rays are black (the shadow). Surface
// brightness conservation makes the direct lookup exact (PHYSICS.md §8).

#include "../common/uniforms.wgsl"
#include "../common/ray-state.wgsl"

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(5) var hdrOut: texture_storage_2d<rgba16float, write>;

const GRID_STEP: f32 = 0.2617993877991494; // 15 degrees
const GRID_LINE_HALF_WIDTH: f32 = 0.0025;  // radians on the sky
const MAX_STEPS_DEBUG_COLOR = vec3f(0.25, 0.0, 0.25);

fn gridLine(angularDistance: f32) -> f32 {
  return 1.0 - smoothstep(0.0, GRID_LINE_HALF_WIDTH, angularDistance);
}

fn skyRadiance(dir: vec3f) -> vec3f {
  let az = atan2(dir.y, dir.x);
  let el = asin(clamp(dir.z, -1.0, 1.0));

  // Angular distance to the nearest graticule line (azimuth scaled to true angle).
  let azDist = abs(fract(az / GRID_STEP + 0.5) - 0.5) * GRID_STEP * max(abs(cos(el)), 0.02);
  let elDist = abs(fract(el / GRID_STEP + 0.5) - 0.5) * GRID_STEP;
  let lines = max(gridLine(azDist), gridLine(elDist)) * 0.10;

  // Deep-sky base plus a soft "galactic" band along the equator for orientation.
  let band = exp(-(el * el) / (0.18 * 0.18));
  var color = vec3f(0.010, 0.013, 0.025) + band * vec3f(0.050, 0.042, 0.034) + vec3f(lines);

  // Antipodal source disc -> Einstein ring when the camera faces the hole.
  let markerCos = dot(dir, u.markerDir.xyz);
  color += smoothstep(u.markerDir.w, u.camPos.w, markerCos) * vec3f(4.0, 4.0, 3.6);
  return color;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn cs_shade(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.resolution.x || gid.y >= u.resolution.y) {
    return;
  }
  let i = gid.y * u.resolution.x + gid.x;

  var color = vec3f(0.0); // CAPTURED (shadow) and still-MARCHING pixels
  switch (status[i]) {
    case STATUS_ESCAPED: {
      let dir = vec3f(
        result[0u * u.rayCount + i],
        result[1u * u.rayCount + i],
        result[2u * u.rayCount + i],
      );
      color = skyRadiance(dir);
    }
    case STATUS_MAX_STEPS: {
      color = MAX_STEPS_DEBUG_COLOR; // diagnostic, counted by V-O3 in M6
    }
    default: {}
  }
  textureStore(hdrOut, gid.xy, vec4f(color, 1.0));
}
