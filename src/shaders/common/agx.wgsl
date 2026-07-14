// AgX picture formation (D10), Rec.2020 working space.
// Port of the Filament/three.js implementation of Sobotka's AgX [R16]:
// 6th-order sigmoid approximation of the AgX contrast curve.
// Input: linear Rec.709 radiance (exposure already applied).
// Output: linear Rec.709 display values in [0, 1]; caller applies the sRGB OETF.

const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

const LINEAR_SRGB_TO_REC2020 = mat3x3f(
  vec3f(0.6274, 0.0691, 0.0164),
  vec3f(0.3293, 0.9195, 0.0880),
  vec3f(0.0433, 0.0113, 0.8956),
);

const REC2020_TO_LINEAR_SRGB = mat3x3f(
  vec3f(1.6605, -0.1246, -0.0182),
  vec3f(-0.5876, 1.1329, -0.1006),
  vec3f(-0.0728, -0.0083, 1.1187),
);

const AGX_INSET = mat3x3f(
  vec3f(0.856627153315983, 0.137318972929847, 0.11189821299995),
  vec3f(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
  vec3f(0.0482516061458583, 0.101439036467562, 0.811302368396859),
);

const AGX_OUTSET = mat3x3f(
  vec3f(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
  vec3f(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
  vec3f(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
);

fn agx_contrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
       + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx_tonemap(linear_srgb: vec3f) -> vec3f {
  var c = LINEAR_SRGB_TO_REC2020 * linear_srgb;
  c = AGX_INSET * c;
  c = max(c, vec3f(1e-10));
  c = clamp((log2(c) - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV), vec3f(0.0), vec3f(1.0));
  c = agx_contrast(c);
  c = pow(max(c, vec3f(0.0)), vec3f(2.2)); // AgX curve is encoded ~2.2; back to linear
  c = REC2020_TO_LINEAR_SRGB * c;
  return clamp(c, vec3f(0.0), vec3f(1.0));
}

fn srgb_oetf(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}
