/**
 * Mean SSIM (Wang et al. 2004) on grayscale images — the V-I5 perceptual
 * metric (VALIDATION.md §3, automated in M8). Standard parameters: 11×11
 * Gaussian window (σ = 1.5), K1 = 0.01, K2 = 0.03, dynamic range L = 255;
 * the score is the mean over all fully-covered (valid-convolution) windows.
 *
 * Pure float64 TypeScript with no DOM dependencies: unit-tested headless in
 * Vitest (tests/unit/validation/ssim.test.ts) and imported by the dev-build
 * browser harness for the golden comparison. Never part of the production
 * bundle (ARCHITECTURE.md §2).
 */

const WINDOW_RADIUS = 5; // 11×11
const SIGMA = 1.5;
const K1 = 0.01;
const K2 = 0.03;
const L = 255;
const C1 = (K1 * L) * (K1 * L);
const C2 = (K2 * L) * (K2 * L);

function gaussianKernel(): Float64Array {
  const size = 2 * WINDOW_RADIUS + 1;
  const k = new Float64Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const d = i - WINDOW_RADIUS;
    k[i] = Math.exp(-(d * d) / (2 * SIGMA * SIGMA));
    sum += k[i]!;
  }
  for (let i = 0; i < size; i++) k[i] = k[i]! / sum;
  return k;
}

const KERNEL = gaussianKernel();

/**
 * Separable valid-region Gaussian filter: (w, h) → (w − 2R, h − 2R).
 * Two inputs are filtered per call where provided (shared row pass cost is
 * negligible next to clarity; each field is filtered independently).
 */
function blurValid(src: Float64Array, width: number, height: number): Float64Array {
  const R = WINDOW_RADIUS;
  const wOut = width - 2 * R;
  const hOut = height - 2 * R;
  // Horizontal pass: (width, height) → (wOut, height).
  const tmp = new Float64Array(wOut * height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < wOut; x++) {
      let acc = 0;
      for (let k = 0; k <= 2 * R; k++) acc += KERNEL[k]! * src[row + x + k]!;
      tmp[y * wOut + x] = acc;
    }
  }
  // Vertical pass: (wOut, height) → (wOut, hOut).
  const out = new Float64Array(wOut * hOut);
  for (let y = 0; y < hOut; y++) {
    for (let x = 0; x < wOut; x++) {
      let acc = 0;
      for (let k = 0; k <= 2 * R; k++) acc += KERNEL[k]! * tmp[(y + k) * wOut + x]!;
      out[y * wOut + x] = acc;
    }
  }
  return out;
}

export interface SsimResult {
  /** Mean SSIM over valid windows. */
  mean: number;
  /** Worst single-window SSIM and its window-center pixel (localization aid). */
  min: number;
  minX: number;
  minY: number;
  windows: number;
}

/**
 * Mean SSIM between two grayscale images in [0, 255]. Images must be at
 * least 11×11 and identically sized.
 */
export function computeSsim(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  width: number,
  height: number,
): SsimResult {
  if (a.length !== width * height || b.length !== width * height) {
    throw new Error(`ssim: image size mismatch (${a.length}, ${b.length} vs ${width}×${height})`);
  }
  const R = WINDOW_RADIUS;
  if (width < 2 * R + 1 || height < 2 * R + 1) throw new Error('ssim: image smaller than the 11×11 window');

  const n = width * height;
  const fa = new Float64Array(n);
  const fb = new Float64Array(n);
  const faa = new Float64Array(n);
  const fbb = new Float64Array(n);
  const fab = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const va = a[i]!;
    const vb = b[i]!;
    fa[i] = va;
    fb[i] = vb;
    faa[i] = va * va;
    fbb[i] = vb * vb;
    fab[i] = va * vb;
  }

  const muA = blurValid(fa, width, height);
  const muB = blurValid(fb, width, height);
  const muAA = blurValid(faa, width, height);
  const muBB = blurValid(fbb, width, height);
  const muAB = blurValid(fab, width, height);

  const wOut = width - 2 * R;
  const hOut = height - 2 * R;
  let sum = 0;
  let min = Infinity;
  let minX = -1;
  let minY = -1;
  for (let y = 0; y < hOut; y++) {
    for (let x = 0; x < wOut; x++) {
      const i = y * wOut + x;
      const ma = muA[i]!;
      const mb = muB[i]!;
      const varA = muAA[i]! - ma * ma;
      const varB = muBB[i]! - mb * mb;
      const covAB = muAB[i]! - ma * mb;
      const s =
        ((2 * ma * mb + C1) * (2 * covAB + C2)) /
        ((ma * ma + mb * mb + C1) * (varA + varB + C2));
      sum += s;
      if (s < min) {
        min = s;
        minX = x + R;
        minY = y + R;
      }
    }
  }
  const windows = wOut * hOut;
  return { mean: sum / windows, min, minX, minY, windows };
}

/** rgba8 bytes → Rec. 709 luma on the encoded (display) values, in [0, 255]. */
export function rgba8ToLuma(bytes: Uint8Array, width: number, height: number): Float64Array {
  if (bytes.length !== width * height * 4) throw new Error('rgba8ToLuma: byte length mismatch');
  const out = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    out[i] = 0.2126 * bytes[i * 4]! + 0.7152 * bytes[i * 4 + 1]! + 0.0722 * bytes[i * 4 + 2]!;
  }
  return out;
}
