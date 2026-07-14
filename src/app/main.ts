/**
 * Bootstrap: acquire GPU -> run render loop -> rebuild everything on device
 * loss (D6: device loss is expected on Windows, not exceptional).
 */
import { acquireGpu, WebGpuUnavailableError } from '../gpu/device';
import { Renderer } from '../renderer/renderer';
import { OrbitCamera } from '../scene/orbitCamera';
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
  const localRenderer = new Renderer(gpu, w, h);
  renderer = localRenderer;

  const camera = new OrbitCamera(() => localRenderer.setCamera(camera.pose));
  camera.attach(canvas);
  localRenderer.setCamera(camera.pose);

  new ResizeObserver(() => {
    const [rw, rh] = canvasPixelSize();
    canvas.width = rw;
    canvas.height = rh;
    localRenderer.resize(rw, rh);
    localRenderer.setCamera(camera.pose); // aspect changed
  }).observe(canvas);

  let lastHudUpdate = 0;
  const frame = (now: number): void => {
    if (!renderer) return;
    renderer.renderFrame();
    if (now - lastHudUpdate > 250) {
      const marching = localRenderer.tracePass.getKnownMarching();
      hud.textContent =
        `LensForge M2 — ${gpu.tier} tier — r=${camera.pose.r.toFixed(1)}M` +
        `${marching > 0 ? ` — resolving ${marching} rays` : ''}\n${renderer.profiler.report()}`;
      lastHudUpdate = now;
    }
    rafHandle = requestAnimationFrame(frame);
  };
  rafHandle = requestAnimationFrame(frame);

  if (import.meta.env.DEV) {
    // Dev-only hooks: device-loss drill, rAF-independent rendering (hidden-tab
    // validation), and the V-O GPU-vs-oracle harness (src/validation).
    (window as unknown as Record<string, unknown>).__lensforge = {
      simulateDeviceLoss: () => gpu.device.destroy(),
      forceRecover: () => {
        cancelAnimationFrame(rafHandle);
        renderer = null;
        void start();
      },
      profilerReport: () => localRenderer.profiler.report(),
      tracePass: localRenderer.tracePass,
      resize: (w: number, h: number) => {
        canvas.width = w;
        canvas.height = h;
        localRenderer.resize(w, h);
        localRenderer.setCamera(camera.pose);
      },
      setCamera: (pose: Partial<import('../renderer/tracePass').CameraPose>) => camera.set(pose),
      renderOnce: async () => {
        localRenderer.renderFrame();
        await gpu.device.queue.onSubmittedWorkDone();
      },
      /** Render frames until the march converges; returns frames used. */
      converge: async (maxFrames = 200) => {
        for (let i = 0; i < maxFrames; i++) {
          localRenderer.renderFrame();
          await gpu.device.queue.onSubmittedWorkDone();
          await new Promise((r) => setTimeout(r, 0)); // let counter mapAsync settle
          if (localRenderer.tracePass.getKnownMarching() === 0) return i + 1;
        }
        return maxFrames;
      },
      runValidation: async () => {
        const { runVOSuite } = await import('../validation/gpuOracleHarness');
        return runVOSuite(localRenderer.tracePass);
      },
      measureShadow: async () => {
        const { measureShadowDiameter } = await import('../validation/gpuOracleHarness');
        return measureShadowDiameter(localRenderer, camera);
      },
    };
  }
}

void start();
