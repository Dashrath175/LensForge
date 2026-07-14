// Pass D (present-only skeleton, M0): HDR target -> exposure -> AgX -> sRGB -> swapchain.
// CPU counterpart: none (pure display transform; radiometry stays physical until here).

#include "../common/agx.wgsl"

struct PresentUniforms {
  exposure: f32,
}

@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: PresentUniforms;

struct VsOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
  // Single fullscreen triangle: (-1,-1) (3,-1) (-1,3)
  var out: VsOut;
  let x = f32(i32(i & 1u) * 4 - 1);
  let y = f32(i32(i >> 1u) * 4 - 1);
  out.pos = vec4f(x, y, 0.0, 1.0);
  return out;
}

@fragment
fn fs_present(in: VsOut) -> @location(0) vec4f {
  let dims = textureDimensions(hdrTex);
  let px = vec2u(clamp(vec2i(in.pos.xy), vec2i(0), vec2i(dims) - 1));
  let hdr = textureLoad(hdrTex, px, 0).rgb;
  let display = agx_tonemap(hdr * u.exposure);
  return vec4f(srgb_oetf(display), 1.0);
}
