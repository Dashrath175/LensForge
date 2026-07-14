/**
 * Bootstrap: acquire GPU -> run render loop -> rebuild everything on device
 * loss (D6: device loss is expected on Windows, not exceptional).
 */
import { acquireGpu, WebGpuUnavailableError } from '../gpu/device';
import { Renderer } from '../renderer/renderer';
import { showCapabilityPage } from './capabilityPage';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const hud = document.getElementById('hud') as HTMLDivElement;

let renderer: Renderer | null = null;
let rafHandle = 0;

function canvasPixelSize(): [number, number] {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return [Math.max(1, Math.floor(canvas.clientWidth * dpr)), Math.max(1, Math.floor(canvas.clientHeight * dpr))];
}

async function start(): Promise<void> {
  let gpu;
  try {
    gpu = await acquireGpu(canvas, (info) => {
      // TDR or driver reset: tear down and rebuild the full GPU state.
      console.warn(`LensForge: GPU device lost (${info.reason ?? 'unknown'}): ${info.message} — recovering.`);
      cancelAnimationFrame(rafHandle);
      renderer = null;
      void start();
    });
  } catch (err) {
    if (err instanceof WebGpuUnavailableError) {
      showCapabilityPage(canvas, err.message);
      return;
    }
    throw err;
  }

  const [w, h] = canvasPixelSize();
  canvas.width = w;
  canvas.height = h;
  renderer = new Renderer(gpu, w, h);

  new ResizeObserver(() => {
    const [rw, rh] = canvasPixelSize();
    canvas.width = rw;
    canvas.height = rh;
    renderer?.resize(rw, rh);
  }).observe(canvas);

  let lastHudUpdate = 0;
  const frame = (now: number): void => {
    if (!renderer) return;
    renderer.renderFrame();
    if (now - lastHudUpdate > 250) {
      hud.textContent = `LensForge M0 — ${gpu.tier} tier\n${renderer.profiler.report()}`;
      lastHudUpdate = now;
    }
    rafHandle = requestAnimationFrame(frame);
  };
  rafHandle = requestAnimationFrame(frame);

  if (import.meta.env.DEV) {
    // Dev-only hooks: device-lost recovery drill and rAF-independent single-frame
    // render (lets headless/hidden-tab validation drive the pipeline directly).
    (window as unknown as Record<string, unknown>).__lensforge = {
      simulateDeviceLoss: () => gpu.device.destroy(),
      forceRecover: () => {
        cancelAnimationFrame(rafHandle);
        renderer = null;
        void start();
      },
      profilerReport: () => renderer?.profiler.report() ?? 'no renderer',
      renderOnce: async (width: number, height: number) => {
        if (!renderer) return 'no renderer';
        canvas.width = width;
        canvas.height = height;
        renderer.resize(width, height);
        renderer.renderFrame();
        await gpu.device.queue.onSubmittedWorkDone();
        return `rendered ${width}x${height}`;
      },
    };
  }
}

void start();
