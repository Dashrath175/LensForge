/**
 * Minimal dependency-free PNG encoder (M8 figure/golden tooling): rgba8 →
 * 8-bit truecolor+alpha PNG, filter type 0, single zlib-deflated IDAT.
 * Node-only (uses node:zlib); shared by tools/generateFigures.ts.
 */
import { deflateSync } from 'node:zlib';

const CRC_TABLE = ((): Uint32Array => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(...parts: Uint8Array[]): number {
  let c = 0xffffffff;
  for (const p of parts) {
    for (let i = 0; i < p.length; i++) c = CRC_TABLE[(c ^ p[i]!) & 0xff]! ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Buffer {
  const typeBytes = Buffer.from(type, 'latin1');
  const head = Buffer.alloc(4);
  head.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(typeBytes, data), 0);
  return Buffer.concat([head, typeBytes, data, crc]);
}

/** Encode tightly-packed rgba8 rows (length = width·height·4) as a PNG. */
export function encodePng(rgba: Uint8Array, width: number, height: number): Buffer {
  if (rgba.length !== width * height * 4) {
    throw new Error(`encodePng: expected ${width * height * 4} bytes, got ${rgba.length}`);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type: truecolor + alpha
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter method
  ihdr[12] = 0; // interlace

  // Filter byte 0 (None) per row.
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0;
    raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', new Uint8Array(0)),
  ]);
}

/** Crop a tightly-packed rgba8 image (clamped to bounds). */
export function cropRgba(
  rgba: Uint8Array,
  width: number,
  height: number,
  x0: number,
  y0: number,
  cw: number,
  ch: number,
): { rgba: Uint8Array; width: number; height: number; x0: number; y0: number } {
  const cx0 = Math.max(0, Math.min(width - 1, Math.round(x0)));
  const cy0 = Math.max(0, Math.min(height - 1, Math.round(y0)));
  const cx1 = Math.max(cx0 + 1, Math.min(width, cx0 + Math.round(cw)));
  const cy1 = Math.max(cy0 + 1, Math.min(height, cy0 + Math.round(ch)));
  const w = cx1 - cx0;
  const h = cy1 - cy0;
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    out.set(rgba.subarray(((cy0 + y) * width + cx0) * 4, ((cy0 + y) * width + cx1) * 4), y * w * 4);
  }
  return { rgba: out, width: w, height: h, x0: cx0, y0: cy0 };
}
