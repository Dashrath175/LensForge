// Pass D: HDR target -> optional bloom composite (M7, camera model) ->
// exposure -> AgX -> sRGB -> swapchain (or the export rgba8 target).
// CPU counterpart: none (pure display transform; radiometry stays physical
// until here). Bloom is an energy-conserving mix with the blur pyramid
// (shaders/post/bloom.wgsl); strength 0 (default) reproduces the pre-M7
// output exactly.

#include "../common/agx.wgsl"

struct PresentUniforms {
  exposure: f32,
  bloomStrength: f32,
  /** 1 / bloom level count: the pyramid accumulates one unit-gain blur per level. */
  bloomNorm: f32,
}

@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: PresentUniforms;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var presentSampler: sampler;

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
  var px = vec2u(clamp(vec2i(in.pos.xy), vec2i(0), vec2i(dims) - 1));
  // Arriving-photon display parity (M7, VALIDATION.md §6): the HDR target
  // holds the traced-photon image, whose physical counterpart is its exact
  // mirror in screen x (Kerr (t, φ) → (−t, −φ) isometry — see sky.wgsl).
  // Statuses/HDR readbacks keep the traced orientation; only display and
  // PNG export (this shader) present the physical arriving-photon image.
  px.x = dims.x - 1u - px.x;
  var hdr = textureLoad(hdrTex, px, 0).rgb;
  if (u.bloomStrength > 0.0) {
    let uv = (vec2f(px) + 0.5) / vec2f(dims);
    let bloom = textureSampleLevel(bloomTex, presentSampler, uv, 0.0).rgb * u.bloomNorm;
    hdr = mix(hdr, bloom, u.bloomStrength);
  }
  let display = agx_tonemap(hdr * u.exposure);
  return vec4f(srgb_oetf(display), 1.0);
}
