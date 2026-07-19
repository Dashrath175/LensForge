/**
 * M7 export (IMPLEMENTATION.md M7): PNG screenshot and tiled high-resolution
 * export (to 8K) through the existing chunk scheduler — the SAME Pass A-C
 * code path as interactive rendering (ARCHITECTURE.md §5: one scheduler for
 * interactive frames, refinement, and tiled export).
 *
 * Tiling: the full image is rendered as TILE×TILE windows via the tile words
 * in the trace uniforms (init.wgsl computes NDC in full-image pixel space, so
 * the geodesics are exactly the ones a monolithic render would trace). Each
 * tile carries a 1-pixel apron: the star-splat Jacobian is finite-differenced
 * from neighbor rays, and the apron gives interior tile borders the same
 * central differences a full-frame render uses — tiles are pixel-consistent
 * with the interactive image by construction. Every tile converges its base
 * trace, then accumulates EXPORT_SAMPLES temporal-AA rounds (the M6
 * machinery), then runs the present transform (exposure → AgX → sRGB) into
 * rgba8 and is assembled into the PNG.
 *
 * TDR safety: nothing changes versus interactive rendering — every dispatch
 * is bounded at STEPS_PER_DISPATCH DP45 attempts (V-P4), and the exporter
 * awaits queue completion between frames.
 *
 * Bloom: forced off for tiled export (blur kernels cross tile borders); the
 * screenshot path captures the presented frame bloom-and-all. Documented in
 * VALIDATION.md §7.
 */
import type { ObserverType } from '../core/parameters';
import type { Renderer } from './renderer';
import type { CameraPose } from './tracePass';

const TILE = 1024;
const APRON = 1;
/** Temporal-AA samples accumulated per tile (base + 7 jittered rounds). */
export const EXPORT_SAMPLES = 8;
/** Per-tile frame guard: a tile that cannot converge in this many frames
 * aborts the export instead of hanging (never observed; defensive). */
const MAX_FRAMES_PER_TILE = 20000;

export interface ExportRequest {
  width: number;
  height: number;
  pose: CameraPose;
  spin: number;
  observer: ObserverType;
  samples?: number;
}

function downloadBlob(blob: Blob, filename: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function encodePng(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('export: 2d context unavailable for PNG encode');
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

export class ImageExporter {
  private readonly renderer: Renderer;
  private busy = false;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
  }

  get exporting(): boolean {
    return this.busy;
  }

  /** PNG of the currently presented frame (canvas resolution, bloom included). */
  async screenshot(filename = 'lensforge.png'): Promise<void> {
    const { bytes, width, height } = await this.renderer.presentToBytes();
    const rgba = new Uint8ClampedArray(new ArrayBuffer(bytes.byteLength));
    rgba.set(bytes);
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
    downloadBlob(await encodePng(rgba, width, height), filename);
  }

  /**
   * Tiled high-resolution export. The caller (src/app/main.ts) must pause
   * interactive rendering and store→renderer application for the duration,
   * and re-apply the store state afterwards (resize + camera restore).
   */
  async exportPng(
    req: ExportRequest,
    onProgress: (tilesDone: number, tilesTotal: number) => void,
    filename?: string,
  ): Promise<void> {
    if (this.busy) throw new Error('export already in progress');
    this.busy = true;
    const renderer = this.renderer;
    const tp = renderer.tracePass;
    const { width: fullW, height: fullH } = req;
    const samples = req.samples ?? EXPORT_SAMPLES;
    const saved = renderer.dimensions();

    const cols = Math.ceil(fullW / TILE);
    const rows = Math.ceil(fullH / TILE);
    const total = cols * rows;
    const out = new Uint8ClampedArray(new ArrayBuffer(fullW * fullH * 4));

    try {
      tp.setPreviewsEnabled(false);
      renderer.setBloomStrength(0); // tiled export: bloom kernels cross tile edges
      renderer.resize(TILE + 2 * APRON, TILE + 2 * APRON);

      let done = 0;
      onProgress(0, total);
      for (let ty = 0; ty < rows; ty++) {
        for (let tx = 0; tx < cols; tx++) {
          const ox = tx * TILE;
          const oy = ty * TILE;
          tp.setCamera(req.pose, req.spin, req.observer, {
            x: ox - APRON,
            y: oy - APRON,
            fullWidth: fullW,
            fullHeight: fullH,
          });

          let frames = 0;
          for (; frames < MAX_FRAMES_PER_TILE; frames++) {
            renderer.renderHeadless();
            await renderer.onGpuIdle();
            await new Promise((r) => setTimeout(r, 0)); // let counter mapAsync settle
            if (tp.getKnownMarching() === 0 && tp.getRefinementState().samples >= samples) break;
          }
          if (frames >= MAX_FRAMES_PER_TILE) {
            throw new Error(`export: tile (${tx}, ${ty}) did not converge`);
          }

          const { bytes, width: tw } = await renderer.presentToBytes();
          const copyW = Math.min(TILE, fullW - ox);
          const copyH = Math.min(TILE, fullH - oy);
          for (let y = 0; y < copyH; y++) {
            const srcRow = ((y + APRON) * tw + APRON) * 4;
            const dstRow = ((oy + y) * fullW + ox) * 4;
            out.set(bytes.subarray(srcRow, srcRow + copyW * 4), dstRow);
          }
          done++;
          onProgress(done, total);
        }
      }
      for (let i = 3; i < out.length; i += 4) out[i] = 255;
      const blob = await encodePng(out, fullW, fullH);
      downloadBlob(blob, filename ?? `lensforge-${fullW}x${fullH}.png`);
    } finally {
      tp.setPreviewsEnabled(true);
      renderer.resize(saved.width, saved.height);
      this.busy = false;
    }
  }
}
