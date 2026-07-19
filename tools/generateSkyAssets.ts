/**
 * Production sky pipeline (IMPLEMENTATION.md M7, D9): NASA panorama →
 * KTX2 cubemap + analytic bright-star catalog + aggregate star cubemap.
 *
 * Physical design (ARCHITECTURE.md D9): the cubemap carries only *extended*
 * radiance (diffuse Milky Way, zodiacal glow, unresolved star field) — its
 * lookup by asymptotic direction is exact under lensing by surface-brightness
 * conservation. Point sources can never be carried by a texture (resolution
 * cannot fix aliasing under strong distortion), so bright catalog stars are
 * measured photometrically IN the panorama, subtracted from it, and shipped
 * as an analytic catalog the renderer splats with |det J|⁻¹ magnification.
 * Because star fluxes are measured from the same panorama the background
 * comes from, splats and cubemap share one radiometric unit system by
 * construction — no cross-calibration constant exists to get wrong. Stars too
 * faint to measure above the local background stay baked in the cubemap:
 * total energy is conserved, nothing is double-counted.
 *
 * Outputs (assets/sky/, shipped):
 *   sky-cubemap.ktx2   6×FACE² rgba16f, star-subtracted background radiance,
 *                      normalized so the sphere-mean radiance = 1
 *   star-agg.ktx2      6×AGG_FACE² rgba16f, catalog-star radiance aggregated
 *                      per texel — the surface-brightness-correct limit used
 *                      when a pixel's sky footprint exceeds the splat regime
 *   bright-stars.bin   f32 records [dirx,diry,dirz, R,G,B flux (radiance·sr), 0,0]
 *                      sorted by (theta,phi) grid cell, + bright-stars.json
 *                      metadata with CSR cell offsets
 *
 * The equirect mapping (u,v) ↔ (RA,Dec) of the source EXR is *calibrated*,
 * not assumed: all 8 axis conventions are scored by total aperture excess of
 * the brightest catalog stars, and the winner must be decisive. This also
 * pins the sky's parity (as seen from inside the celestial sphere).
 */
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import {
  KHR_DF_CHANNEL_RGBSDA_ALPHA,
  KHR_DF_SAMPLE_DATATYPE_FLOAT,
  KHR_DF_SAMPLE_DATATYPE_LINEAR,
  KHR_DF_SAMPLE_DATATYPE_SIGNED,
  KHR_DF_TRANSFER_LINEAR,
  VK_FORMAT_R16G16B16A16_SFLOAT,
  write as writeKtx,
  type KTX2Container,
} from 'ktx-parse';
import parseExr from 'parse-exr';
import { fetchSkySources } from './fetchSkySources';

// ---------------------------------------------------------------------------
// Parameters (all documented in bright-stars.json for reproducibility).
// ---------------------------------------------------------------------------

/** Shipped cubemap face size. Background is smooth post-subtraction; stars —
 * the only content needing resolution — are analytic splats (D9). */
const FACE = 512;
/** Aggregate star-radiance cubemap face size (demagnified-regime fallback). */
const AGG_FACE = 64;
/** Supersamples per texel axis when resampling equirect → cube face. */
const SUPER = 4;
/** Catalog magnitude cut: BSC5 is complete to V ≈ 6.5 (naked-eye sky). */
const VMAG_MAX = 6.5;
/** Aperture photometry radii in source-panorama pixels. */
const APERTURE_R = 3.0;
const ANNULUS_R0 = 5.0;
const ANNULUS_R1 = 8.0;
/** Star-splat spatial index: theta × phi cells (mirrored in sky.wgsl). */
const GRID_NTH = 32;
const GRID_NPH = 64;

// ---------------------------------------------------------------------------
// Float16 encoding.
// ---------------------------------------------------------------------------

function f32ToF16(v: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = v;
  const x = u32[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let frac = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (frac ? 0x200 : 0); // inf/NaN
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00; // overflow → inf
  if (e <= 0) {
    if (e < -10) return sign; // underflow → 0
    frac |= 0x800000;
    const shift = 14 - e;
    return sign | ((frac + (1 << (shift - 1))) >>> shift);
  }
  return sign | (e << 10) | ((frac + 0x1000) >>> 13);
}

// ---------------------------------------------------------------------------
// Equirect panorama with a calibrated (u,v) ↔ (RA,Dec) mapping.
// ---------------------------------------------------------------------------

interface Panorama {
  width: number;
  height: number;
  /** RGB triplets, row-major from data row 0 (whatever orientation that is). */
  data: Float32Array;
}

interface Mapping {
  /** u = frac(uSign * RA/2π + uOff) */
  uSign: 1 | -1;
  uOff: number;
  /** v = 0.5 + vSign * (−Dec/π) — vSign=+1 puts Dec=+90° at v=0 (row 0). */
  vSign: 1 | -1;
}

function mapUv(m: Mapping, raRad: number, decRad: number): [number, number] {
  let u = m.uSign * (raRad / (2 * Math.PI)) + m.uOff;
  u -= Math.floor(u);
  const v = 0.5 + m.vSign * (-decRad / Math.PI);
  return [u, v];
}

/** Bilinear sample with RA wrap in u and clamp in v. Returns RGB. */
function sampleEquirect(p: Panorama, u: number, v: number, out: [number, number, number]): void {
  const x = u * p.width - 0.5;
  const y = v * p.height - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const xi0 = ((x0 % p.width) + p.width) % p.width;
  const xi1 = (xi0 + 1) % p.width;
  const yi0 = Math.min(Math.max(y0, 0), p.height - 1);
  const yi1 = Math.min(Math.max(y0 + 1, 0), p.height - 1);
  for (let c = 0; c < 3; c++) {
    const a = p.data[(yi0 * p.width + xi0) * 3 + c]! * (1 - fx) + p.data[(yi0 * p.width + xi1) * 3 + c]! * fx;
    const b = p.data[(yi1 * p.width + xi0) * 3 + c]! * (1 - fx) + p.data[(yi1 * p.width + xi1) * 3 + c]! * fx;
    out[c] = a * (1 - fy) + b * fy;
  }
}

// ---------------------------------------------------------------------------
// BSC5 catalog.
// ---------------------------------------------------------------------------

interface CatalogStar {
  hr: number;
  name: string;
  raRad: number;
  decRad: number;
  vmag: number;
}

function parseBsc5(raw: Buffer): CatalogStar[] {
  const text = gunzipSync(raw).toString('latin1');
  const stars: CatalogStar[] = [];
  for (const line of text.split('\n')) {
    if (line.length < 107) continue;
    const raH = parseFloat(line.slice(75, 77));
    const raM = parseFloat(line.slice(77, 79));
    const raS = parseFloat(line.slice(79, 83));
    const deSign = line[83] === '-' ? -1 : 1;
    const deD = parseFloat(line.slice(84, 86));
    const deM = parseFloat(line.slice(86, 88));
    const deS = parseFloat(line.slice(88, 90));
    const vmag = parseFloat(line.slice(102, 107));
    if (![raH, raM, raS, deD, deM, deS, vmag].every(Number.isFinite)) continue; // novae etc.
    if (vmag > VMAG_MAX) continue;
    stars.push({
      hr: parseInt(line.slice(0, 4), 10),
      name: line.slice(4, 14).trim(),
      raRad: ((raH + raM / 60 + raS / 3600) / 24) * 2 * Math.PI,
      decRad: (deSign * (deD + deM / 60 + deS / 3600) * Math.PI) / 180,
      vmag,
    });
  }
  return stars;
}

// ---------------------------------------------------------------------------
// Mapping calibration: score candidate axis conventions by aperture excess.
// ---------------------------------------------------------------------------

function apertureExcess(
  p: Panorama,
  u: number,
  v: number,
  perChannel: [number, number, number] | null,
): number {
  const cx = u * p.width - 0.5;
  const cy = v * p.height - 0.5;
  const cxi = Math.round(cx);
  const cyi = Math.round(cy);
  const r1 = Math.ceil(ANNULUS_R1);
  if (cyi < r1 || cyi >= p.height - r1) return 0; // pole-adjacent: skip
  const chan: [number, number, number] = [0, 0, 0];
  const annulus: Array<[number, number, number]> = [];
  let nAp = 0;
  for (let dy = -r1; dy <= r1; dy++) {
    const yy = cyi + dy;
    for (let dx = -r1; dx <= r1; dx++) {
      const xx = (((cxi + dx) % p.width) + p.width) % p.width;
      const rr = Math.hypot(cxi + dx - cx, yy - cy);
      const idx = (yy * p.width + xx) * 3;
      if (rr <= APERTURE_R) {
        nAp++;
        for (let c = 0; c < 3; c++) chan[c] = chan[c]! + p.data[idx + c]!;
      } else if (rr >= ANNULUS_R0 && rr <= ANNULUS_R1) {
        annulus.push([p.data[idx]!, p.data[idx + 1]!, p.data[idx + 2]!]);
      }
    }
  }
  const bg: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const vals = annulus.map((s) => s[c]!).sort((a, b) => a - b);
    bg[c] = vals[Math.floor(vals.length / 2)] ?? 0;
  }
  let excess = 0;
  for (let c = 0; c < 3; c++) {
    const e = Math.max(0, chan[c]! - bg[c]! * nAp);
    if (perChannel) perChannel[c] = e;
    excess += e;
  }
  return excess;
}

function calibrateMapping(p: Panorama, stars: CatalogStar[]): { mapping: Mapping; score: number; runnerUp: number } {
  const bright = [...stars].sort((a, b) => a.vmag - b.vmag).slice(0, 200);
  const candidates: Mapping[] = [];
  for (const uSign of [1, -1] as const) {
    for (const uOff of [0, 0.5]) {
      for (const vSign of [1, -1] as const) {
        candidates.push({ uSign, uOff, vSign });
      }
    }
  }
  let best: Mapping = candidates[0]!;
  let bestScore = -1;
  let runnerUp = -1;
  for (const cand of candidates) {
    let score = 0;
    for (const s of bright) {
      const [u, v] = mapUv(cand, s.raRad, s.decRad);
      score += apertureExcess(p, u, v, null);
    }
    if (score > bestScore) {
      runnerUp = bestScore;
      bestScore = score;
      best = cand;
    } else if (score > runnerUp) {
      runnerUp = score;
    }
  }
  return { mapping: best, score: bestScore, runnerUp };
}

// ---------------------------------------------------------------------------
// Star measurement + subtraction.
// ---------------------------------------------------------------------------

interface MeasuredStar {
  hr: number;
  name: string;
  dir: [number, number, number];
  /** Radiant flux per channel, units: [stored radiance] · sr. */
  flux: [number, number, number];
}

/** Pixel solid angle of an equirect panorama at row y (radians²). */
function pixelSolidAngle(p: Panorama, y: number, mapping: Mapping): number {
  const v = (y + 0.5) / p.height;
  const dec = -(v - 0.5) * Math.PI * mapping.vSign;
  return ((2 * Math.PI) / p.width) * (Math.PI / p.height) * Math.cos(dec);
}

/**
 * Measure each catalog star's aperture excess (per channel), then remove it
 * from the panorama by flattening the aperture onto the annulus-median
 * background. Brightest first, so overlapping companions are measured against
 * an already-cleaned neighborhood.
 */
function measureAndSubtract(p: Panorama, stars: CatalogStar[], mapping: Mapping): MeasuredStar[] {
  const ordered = [...stars].sort((a, b) => a.vmag - b.vmag);
  const out: MeasuredStar[] = [];
  const r1 = Math.ceil(ANNULUS_R1);
  for (const s of ordered) {
    const [u, v] = mapUv(mapping, s.raRad, s.decRad);
    const chan: [number, number, number] = [0, 0, 0];
    const excess = apertureExcess(p, u, v, chan);
    if (excess <= 0) continue; // below the map's noise floor: stays baked in
    const cx = u * p.width - 0.5;
    const cy = v * p.height - 0.5;
    const cyi = Math.round(cy);
    if (cyi < r1 || cyi >= p.height - r1) continue;
    // Per-channel annulus median background for the flatten.
    const chanSamples: Array<[number, number, number]> = [];
    for (let dy = -r1; dy <= r1; dy++) {
      const yy = cyi + dy;
      for (let dx = -r1; dx <= r1; dx++) {
        const xx = (((Math.round(cx) + dx) % p.width) + p.width) % p.width;
        const rr = Math.hypot(Math.round(cx) + dx - cx, yy - cy);
        if (rr >= ANNULUS_R0 && rr <= ANNULUS_R1) {
          const idx = (yy * p.width + xx) * 3;
          chanSamples.push([p.data[idx]!, p.data[idx + 1]!, p.data[idx + 2]!]);
        }
      }
    }
    const bg: [number, number, number] = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const vals = chanSamples.map((cs) => cs[c]!).sort((a, b) => a - b);
      bg[c] = vals[Math.floor(vals.length / 2)] ?? 0;
    }
    // Flatten the aperture to background (removes the star and its PSF).
    for (let dy = -r1; dy <= r1; dy++) {
      const yy = cyi + dy;
      for (let dx = -r1; dx <= r1; dx++) {
        const xx = (((Math.round(cx) + dx) % p.width) + p.width) % p.width;
        const rr = Math.hypot(Math.round(cx) + dx - cx, yy - cy);
        if (rr <= APERTURE_R) {
          const idx = (yy * p.width + xx) * 3;
          for (let c = 0; c < 3; c++) p.data[idx + c] = bg[c]!;
        }
      }
    }
    const omega = pixelSolidAngle(p, cyi, mapping);
    const cosD = Math.cos(s.decRad);
    out.push({
      hr: s.hr,
      name: s.name,
      dir: [cosD * Math.cos(s.raRad), cosD * Math.sin(s.raRad), Math.sin(s.decRad)],
      flux: [chan[0]! * omega, chan[1]! * omega, chan[2]! * omega],
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Cube faces (WebGPU/Vulkan order and orientation: +X,−X,+Y,−Y,+Z,−Z).
// ---------------------------------------------------------------------------

function faceDir(face: number, sc: number, tc: number): [number, number, number] {
  switch (face) {
    case 0: return [1, -tc, -sc];
    case 1: return [-1, -tc, sc];
    case 2: return [sc, 1, tc];
    case 3: return [sc, -1, -tc];
    case 4: return [sc, -tc, 1];
    default: return [-sc, -tc, -1];
  }
}

/** Inverse of faceDir: direction → face, sc, tc. */
function dirToFace(d: [number, number, number]): { face: number; sc: number; tc: number } {
  const [x, y, z] = d;
  const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  if (ax >= ay && ax >= az) {
    return x > 0
      ? { face: 0, sc: -z / ax, tc: -y / ax }
      : { face: 1, sc: z / ax, tc: -y / ax };
  }
  if (ay >= ax && ay >= az) {
    return y > 0
      ? { face: 2, sc: x / ay, tc: z / ay }
      : { face: 3, sc: x / ay, tc: -z / ay };
  }
  return z > 0
    ? { face: 4, sc: x / az, tc: -y / az }
    : { face: 5, sc: -x / az, tc: -y / az };
}

function dirToRaDec(d: [number, number, number]): [number, number] {
  const ra = Math.atan2(d[1], d[0]);
  return [ra < 0 ? ra + 2 * Math.PI : ra, Math.asin(Math.min(1, Math.max(-1, d[2])))];
}

/** Resample the (star-subtracted) panorama onto 6 cube faces, RGB f32. */
function buildCubemap(p: Panorama, mapping: Mapping, face: number): Float32Array[] {
  const faces: Float32Array[] = [];
  const sample: [number, number, number] = [0, 0, 0];
  for (let f = 0; f < 6; f++) {
    const data = new Float32Array(face * face * 3);
    for (let j = 0; j < face; j++) {
      for (let i = 0; i < face; i++) {
        let r = 0, g = 0, b = 0;
        for (let sj = 0; sj < SUPER; sj++) {
          for (let si = 0; si < SUPER; si++) {
            const sc = (2 * (i + (si + 0.5) / SUPER)) / face - 1;
            const tc = (2 * (j + (sj + 0.5) / SUPER)) / face - 1;
            const d = faceDir(f, sc, tc);
            const n = Math.hypot(d[0], d[1], d[2]);
            const [ra, dec] = dirToRaDec([d[0] / n, d[1] / n, d[2] / n]);
            const [u, v] = mapUv(mapping, ra, dec);
            sampleEquirect(p, u, v, sample);
            r += sample[0];
            g += sample[1];
            b += sample[2];
          }
        }
        const inv = 1 / (SUPER * SUPER);
        const idx = (j * face + i) * 3;
        data[idx] = r * inv;
        data[idx + 1] = g * inv;
        data[idx + 2] = b * inv;
      }
    }
    faces.push(data);
  }
  return faces;
}

/** Solid angle of a cube-face texel at (sc, tc), face size n (radians²). */
function texelSolidAngle(sc: number, tc: number, n: number): number {
  const du = 2 / n;
  return (du * du) / Math.pow(1 + sc * sc + tc * tc, 1.5);
}

/** Aggregate catalog-star radiance cubemap (RGB f32 faces). */
function buildStarAggregate(stars: MeasuredStar[], face: number): Float32Array[] {
  const faces = Array.from({ length: 6 }, () => new Float32Array(face * face * 3));
  for (const s of stars) {
    const { face: f, sc, tc } = dirToFace(s.dir);
    const i = Math.min(face - 1, Math.max(0, Math.floor(((sc + 1) / 2) * face)));
    const j = Math.min(face - 1, Math.max(0, Math.floor(((tc + 1) / 2) * face)));
    const cSc = (2 * (i + 0.5)) / face - 1;
    const cTc = (2 * (j + 0.5)) / face - 1;
    const omega = texelSolidAngle(cSc, cTc, face);
    const idx = (j * face + i) * 3;
    for (let c = 0; c < 3; c++) faces[f]![idx + c] = faces[f]![idx + c]! + s.flux[c]! / omega;
  }
  return faces;
}

// ---------------------------------------------------------------------------
// KTX2 writer (uncompressed rgba16f cubemap; ktx-parse container).
// ---------------------------------------------------------------------------

function writeCubemapKtx2(path: string, faces: Float32Array[], face: number): string {
  const bytesPerFace = face * face * 4 * 2;
  const level = new Uint8Array(bytesPerFace * 6);
  const view = new DataView(level.buffer);
  for (let f = 0; f < 6; f++) {
    const src = faces[f]!;
    for (let t = 0; t < face * face; t++) {
      const o = f * bytesPerFace + t * 8;
      view.setUint16(o, f32ToF16(src[t * 3]!), true);
      view.setUint16(o + 2, f32ToF16(src[t * 3 + 1]!), true);
      view.setUint16(o + 4, f32ToF16(src[t * 3 + 2]!), true);
      view.setUint16(o + 6, f32ToF16(1), true);
    }
  }
  const samples = [0, 1, 2, KHR_DF_CHANNEL_RGBSDA_ALPHA].map((channelType, c) => ({
    bitOffset: c * 16,
    bitLength: 15,
    channelType:
      channelType |
      KHR_DF_SAMPLE_DATATYPE_FLOAT |
      KHR_DF_SAMPLE_DATATYPE_SIGNED |
      KHR_DF_SAMPLE_DATATYPE_LINEAR,
    samplePosition: [0, 0, 0, 0],
    sampleLower: 0xbf800000,
    sampleUpper: 0x3f800000,
  }));
  const container: KTX2Container = {
    vkFormat: VK_FORMAT_R16G16B16A16_SFLOAT,
    typeSize: 2,
    pixelWidth: face,
    pixelHeight: face,
    pixelDepth: 0,
    layerCount: 0,
    faceCount: 6,
    levelCount: 1,
    supercompressionScheme: 0,
    levels: [{ levelData: level, uncompressedByteLength: level.byteLength }],
    dataFormatDescriptor: [
      {
        vendorId: 0,
        descriptorType: 0,
        versionNumber: 2,
        colorModel: 1, // RGBSDA
        colorPrimaries: 1, // BT709 (linear sRGB primaries)
        transferFunction: KHR_DF_TRANSFER_LINEAR,
        flags: 0,
        texelBlockDimension: [0, 0, 0, 0],
        bytesPlane: [8, 0, 0, 0, 0, 0, 0, 0],
        samples,
      },
    ],
    keyValue: { KTXwriter: 'LensForge tools/generateSkyAssets.ts' },
    globalData: null,
  };
  const bytes = writeKtx(container);
  writeFileSync(path, bytes);
  return createHash('sha256').update(bytes).digest('hex');
}

// ---------------------------------------------------------------------------
// Star grid (CSR over theta × phi cells; mirrored in src/shaders/sky/sky.wgsl).
// ---------------------------------------------------------------------------

function starCell(dir: [number, number, number]): number {
  const theta = Math.acos(Math.min(1, Math.max(-1, dir[2])));
  let phi = Math.atan2(dir[1], dir[0]);
  if (phi < 0) phi += 2 * Math.PI;
  const it = Math.min(GRID_NTH - 1, Math.floor((theta / Math.PI) * GRID_NTH));
  const ip = Math.min(GRID_NPH - 1, Math.floor((phi / (2 * Math.PI)) * GRID_NPH));
  return it * GRID_NPH + ip;
}

// ---------------------------------------------------------------------------
// Self-check: reconstruct random directions from the cubemap vs the source.
// ---------------------------------------------------------------------------

function sampleCubemapCpu(faces: Float32Array[], face: number, d: [number, number, number]): [number, number, number] {
  const { face: f, sc, tc } = dirToFace(d);
  const x = ((sc + 1) / 2) * face - 0.5;
  const y = ((tc + 1) / 2) * face - 0.5;
  const x0 = Math.min(face - 1, Math.max(0, Math.floor(x)));
  const y0 = Math.min(face - 1, Math.max(0, Math.floor(y)));
  const x1 = Math.min(face - 1, x0 + 1);
  const y1 = Math.min(face - 1, y0 + 1);
  const fx = Math.min(1, Math.max(0, x - x0));
  const fy = Math.min(1, Math.max(0, y - y0));
  const src = faces[f]!;
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const a = src[(y0 * face + x0) * 3 + c]! * (1 - fx) + src[(y0 * face + x1) * 3 + c]! * fx;
    const b = src[(y1 * face + x0) * 3 + c]! * (1 - fx) + src[(y1 * face + x1) * 3 + c]! * fx;
    out[c] = a * (1 - fy) + b * fy;
  }
  return out;
}

// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  await fetchSkySources();

  console.log('parsing EXR…');
  const exrBuf = readFileSync(resolve(process.cwd(), 'assets/sky-src/starmap_2020_4k.exr'));
  const FloatType = 1015;
  const exr = parseExr(
    exrBuf.buffer.slice(exrBuf.byteOffset, exrBuf.byteOffset + exrBuf.byteLength),
    FloatType,
  ) as { data: Float32Array; width: number; height: number };
  const { width, height } = exr;
  const channels = exr.data.length / (width * height);
  console.log(`EXR: ${width}×${height}, ${channels} channels`);

  const pano: Panorama = { width, height, data: new Float32Array(width * height * 3) };
  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) pano.data[i * 3 + c] = Math.max(0, exr.data[i * channels + c]!);
  }

  console.log('parsing BSC5…');
  const stars = parseBsc5(readFileSync(resolve(process.cwd(), 'assets/sky-src/bsc5.dat.gz')));
  console.log(`catalog: ${stars.length} stars with V ≤ ${VMAG_MAX}`);

  console.log('calibrating equirect mapping…');
  const { mapping, score, runnerUp } = calibrateMapping(pano, stars);
  console.log(
    `mapping: uSign=${mapping.uSign} uOff=${mapping.uOff} vSign=${mapping.vSign} ` +
      `(score ${score.toExponential(3)}, runner-up ${runnerUp.toExponential(3)})`,
  );
  if (!(score > 2 * runnerUp)) {
    throw new Error('mapping calibration is not decisive — refusing to guess the sky orientation');
  }

  console.log('measuring + subtracting catalog stars…');
  const measured = measureAndSubtract(pano, stars, mapping);
  console.log(`measured ${measured.length}/${stars.length} stars above the map noise floor`);
  const byLum = [...measured].sort(
    (a, b) => b.flux[0] + b.flux[1] + b.flux[2] - (a.flux[0] + a.flux[1] + a.flux[2]),
  );
  console.log('brightest measured stars (sanity: expect Sirius/Canopus/Arcturus/Vega/α Cen…):');
  for (const s of byLum.slice(0, 10)) {
    console.log(
      `  HR ${String(s.hr).padStart(4)} ${s.name.padEnd(10)} flux ${(s.flux[0] + s.flux[1] + s.flux[2]).toExponential(3)}`,
    );
  }

  // Normalize: sphere-mean background radiance (per channel mean of the
  // luminance proxy R+G+B over solid angle) = 1 stored unit.
  let sum = 0;
  let wsum = 0;
  for (let y = 0; y < height; y++) {
    const w = pixelSolidAngle(pano, y, mapping);
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 3;
      sum += (pano.data[idx]! + pano.data[idx + 1]! + pano.data[idx + 2]!) * w;
      wsum += w;
    }
  }
  const mean = sum / wsum / 3;
  const scale = 1 / mean;
  console.log(`normalization: sphere-mean channel radiance ${mean.toExponential(4)} → ×${scale.toExponential(4)}`);
  for (let i = 0; i < pano.data.length; i++) pano.data[i] = pano.data[i]! * scale;
  for (const s of measured) for (let c = 0; c < 3; c++) s.flux[c] = s.flux[c]! * scale;

  // Percentiles for choosing the in-app radiance scale (documented there).
  const lums: number[] = [];
  for (let i = 0; i < width * height; i += 17) {
    lums.push(pano.data[i * 3]! + pano.data[i * 3 + 1]! + pano.data[i * 3 + 2]!);
  }
  lums.sort((a, b) => a - b);
  const pct = (q: number): number => lums[Math.min(lums.length - 1, Math.floor(q * lums.length))]!;
  console.log(
    `background luminance percentiles (R+G+B): p50 ${pct(0.5).toFixed(3)} p90 ${pct(0.9).toFixed(3)} ` +
      `p99 ${pct(0.99).toFixed(3)} p99.9 ${pct(0.999).toFixed(3)} max ${lums[lums.length - 1]!.toFixed(3)}`,
  );

  console.log('building sky cubemap…');
  const skyFaces = buildCubemap(pano, mapping, FACE);
  console.log('building star aggregate…');
  const aggFaces = buildStarAggregate(measured, AGG_FACE);

  // Self-check: cubemap reconstruction vs source panorama on random dirs.
  let rec = 0;
  let recRef = 0;
  const rng = (() => {
    let s = 42;
    return () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  })();
  const sample: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 4000; k++) {
    const z = 2 * rng() - 1;
    const ph = 2 * Math.PI * rng();
    const rxy = Math.sqrt(Math.max(0, 1 - z * z));
    const d: [number, number, number] = [rxy * Math.cos(ph), rxy * Math.sin(ph), z];
    const cube = sampleCubemapCpu(skyFaces, FACE, d);
    const [ra, dec] = dirToRaDec(d);
    const [u, v] = mapUv(mapping, ra, dec);
    sampleEquirect(pano, u, v, sample);
    rec += Math.abs(cube[0] + cube[1] + cube[2] - (sample[0] + sample[1] + sample[2]));
    recRef += sample[0] + sample[1] + sample[2];
  }
  console.log(`cubemap self-check: mean |Δ|/mean = ${(rec / recRef).toExponential(3)} (downsample residual)`);

  mkdirSync(resolve(process.cwd(), 'assets/sky'), { recursive: true });
  const skySha = writeCubemapKtx2(resolve(process.cwd(), 'assets/sky/sky-cubemap.ktx2'), skyFaces, FACE);
  const aggSha = writeCubemapKtx2(resolve(process.cwd(), 'assets/sky/star-agg.ktx2'), aggFaces, AGG_FACE);

  // Star records sorted by grid cell + CSR offsets.
  const cellOf = measured.map((s) => starCell(s.dir));
  const order = measured.map((_, i) => i).sort((a, b) => cellOf[a]! - cellOf[b]!);
  const records = new Float32Array(measured.length * 8);
  const cellStart = new Uint32Array(GRID_NTH * GRID_NPH + 1);
  for (const i of order) cellStart[cellOf[i]! + 1] = cellStart[cellOf[i]! + 1]! + 1;
  for (let c = 0; c < GRID_NTH * GRID_NPH; c++) cellStart[c + 1] = cellStart[c + 1]! + cellStart[c]!;
  order.forEach((srcIdx, k) => {
    const s = measured[srcIdx]!;
    records.set([...s.dir, ...s.flux, 0, 0], k * 8);
  });
  const binPath = resolve(process.cwd(), 'assets/sky/bright-stars.bin');
  const binBytes = Buffer.concat([
    Buffer.from(records.buffer, 0, records.byteLength),
    Buffer.from(cellStart.buffer, 0, cellStart.byteLength),
  ]);
  writeFileSync(binPath, binBytes);
  const starsSha = createHash('sha256').update(binBytes).digest('hex');

  const meta = {
    format: 'f32 records [dirx,diry,dirz,fluxR,fluxG,fluxB,0,0] × starCount, then u32 cellStart × (gridNth*gridNph+1)',
    starCount: measured.length,
    gridNth: GRID_NTH,
    gridNph: GRID_NPH,
    fluxUnits: 'stored-radiance · sr (stored radiance normalized to sphere-mean channel radiance = 1)',
    vmagMax: VMAG_MAX,
    faceSize: FACE,
    aggFaceSize: AGG_FACE,
    mapping,
    sources: {
      panorama:
        'NASA/GSFC Scientific Visualization Studio, Deep Star Maps 2020 (SVS 4851), starmap_2020_4k.exr, public domain',
      catalog:
        'Yale Bright Star Catalogue, 5th revised ed. (Hoffleit & Warren 1991), Harvard-Smithsonian TDC bsc5.dat',
    },
    sha256: { 'sky-cubemap.ktx2': skySha, 'star-agg.ktx2': aggSha, 'bright-stars.bin': starsSha },
  };
  writeFileSync(resolve(process.cwd(), 'assets/sky/bright-stars.json'), `${JSON.stringify(meta, null, 2)}\n`);
  console.log(
    `wrote assets/sky: sky-cubemap.ktx2 (${(6 * FACE * FACE * 8) / 1e6} MB), ` +
      `star-agg.ktx2 (${(6 * AGG_FACE * AGG_FACE * 8) / 1e6} MB), ` +
      `bright-stars.bin (${binBytes.length / 1e3} kB, ${measured.length} stars)`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
