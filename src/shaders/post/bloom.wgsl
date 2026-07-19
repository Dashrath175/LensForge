// M7 optional bloom (ARCHITECTURE.md §4 Pass D, off by default): a dual-filter
// blur pyramid over the HDR target. Documented camera model, not physics: the
// present pass composites mix(hdr, bloom/levels, strength) — an
// energy-conserving redistribution of the already-physical radiance (every
// filter below has unit DC gain), approximating veiling glare of a real
// optical system. No thresholds: glare applies to all light, which is what
// makes the beaming-bright disk edge bleed while the faint sky stays put.

@group(0) @binding(0) var bloomSampler: sampler;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var dstTex: texture_storage_2d<rgba16float, write>;
// Up pass only: same-level downsample result added back (skip connection).
@group(0) @binding(3) var skipTex: texture_2d<f32>;

fn srcSample(uv: vec2f, ts: vec2f, off: vec2f) -> vec3f {
  return textureSampleLevel(srcTex, bloomSampler, uv + ts * off, 0.0).rgb;
}

// 13-tap dual-filter downsample (Jimenez SIGGRAPH 2014); unit DC gain.
@compute @workgroup_size(8, 8)
fn cs_down(@builtin(global_invocation_id) gid: vec3u) {
  let dstDims = textureDimensions(dstTex);
  if (gid.x >= dstDims.x || gid.y >= dstDims.y) {
    return;
  }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dstDims);
  let ts = 1.0 / vec2f(textureDimensions(srcTex));

  let e = srcSample(uv, ts, vec2f(0.0, 0.0));
  let a = srcSample(uv, ts, vec2f(-2.0, -2.0));
  let b = srcSample(uv, ts, vec2f(0.0, -2.0));
  let c = srcSample(uv, ts, vec2f(2.0, -2.0));
  let d = srcSample(uv, ts, vec2f(-2.0, 0.0));
  let f = srcSample(uv, ts, vec2f(2.0, 0.0));
  let g = srcSample(uv, ts, vec2f(-2.0, 2.0));
  let h = srcSample(uv, ts, vec2f(0.0, 2.0));
  let i = srcSample(uv, ts, vec2f(2.0, 2.0));
  let j = srcSample(uv, ts, vec2f(-1.0, -1.0));
  let k = srcSample(uv, ts, vec2f(1.0, -1.0));
  let l = srcSample(uv, ts, vec2f(-1.0, 1.0));
  let m = srcSample(uv, ts, vec2f(1.0, 1.0));

  let color = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  textureStore(dstTex, gid.xy, vec4f(color, 1.0));
}

// 9-tap tent upsample + skip add; unit DC gain per term.
@compute @workgroup_size(8, 8)
fn cs_up(@builtin(global_invocation_id) gid: vec3u) {
  let dstDims = textureDimensions(dstTex);
  if (gid.x >= dstDims.x || gid.y >= dstDims.y) {
    return;
  }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dstDims);
  let ts = 1.0 / vec2f(textureDimensions(srcTex));

  var color = srcSample(uv, ts, vec2f(0.0, 0.0)) * 0.25;
  color += (srcSample(uv, ts, vec2f(-1.0, 0.0)) + srcSample(uv, ts, vec2f(1.0, 0.0)) +
            srcSample(uv, ts, vec2f(0.0, -1.0)) + srcSample(uv, ts, vec2f(0.0, 1.0))) * 0.125;
  color += (srcSample(uv, ts, vec2f(-1.0, -1.0)) + srcSample(uv, ts, vec2f(1.0, -1.0)) +
            srcSample(uv, ts, vec2f(-1.0, 1.0)) + srcSample(uv, ts, vec2f(1.0, 1.0))) * 0.0625;
  let skip = textureLoad(skipTex, vec2i(gid.xy), 0).rgb;
  textureStore(dstTex, gid.xy, vec4f(color + skip, 1.0));
}
