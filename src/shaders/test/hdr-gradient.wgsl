// M0 test pattern (replaced by the trace/shade passes in M2): writes an HDR
// radiance sweep spanning 2^-EV_RANGE .. 2^+EV_RANGE horizontally and hue
// vertically, to exercise the rgba16float target and the AgX present path.

override EV_RANGE: f32 = 8.0;
override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(0) var hdrOut: texture_storage_2d<rgba16float, write>;

fn hue_to_rgb(h: f32) -> vec3f {
  let k = vec3f(h * 6.0) + vec3f(0.0, 4.0, 2.0);
  let m = abs(k % 6.0 - 3.0) - 1.0;
  return clamp(m, vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn cs_pattern(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(hdrOut);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let uv = vec2f(gid.xy) / vec2f(dims);
  let luminance = exp2(mix(-EV_RANGE, EV_RANGE, uv.x));
  // Bottom band: neutral gray sweep (tonemap neutrality check); above: hue sweep.
  var color = vec3f(1.0);
  if (uv.y < 0.85) {
    color = mix(hue_to_rgb(uv.y / 0.85), vec3f(1.0), 0.15);
  }
  textureStore(hdrOut, gid.xy, vec4f(color * luminance, 1.0));
}
