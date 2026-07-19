/**
 * Bootstrap (ARCHITECTURE.md §1): acquire GPU → wire ParameterStore ↔
 * renderer ↔ UI → run the decoupled sim/render loops → rebuild everything on
 * device loss (D6: device loss is expected on Windows, not exceptional).
 *
 * Data flow per ARCHITECTURE.md §6: UI and camera controller write ONLY to
 * the ParameterStore; the store classifies each change (physics → restart
 * accumulation, display → present-pass only, time → sim loop only) and this
 * file applies it to the renderer. The sim loop advances t_sim and, for the
 * circular-orbit observer, the camera's orbital phase — a physics change per
 * tick that the interaction preview presents seamlessly.
 */
import { cloneParameters, DEFAULT_PARAMETERS, type Parameters } from '../core/parameters';
import { APP_VERSION } from '../core/version';
import { angularVelocity } from '../physics/orbits';
import { acquireGpu, WebGpuUnavailableError } from '../gpu/device';
import { loadDiskLuts } from '../renderer/diskLuts';
import { loadSkyAssets } from '../renderer/skyAssets';
import { benchmarkMarchWorkgroupSize } from '../renderer/marchBenchmark';
import { ImageExporter } from '../renderer/export';
import { Renderer } from '../renderer/renderer';
import type { CameraPose, DiskParams } from '../renderer/tracePass';
import { OrbitCamera } from '../scene/orbitCamera';
import { parsePreset, serializePreset, SHIPPED_PRESETS } from '../scene/presets';
import { SimClock } from '../sim/clock';
import { ParameterStore } from '../sim/parameterStore';
import { buildPanel } from '../ui/panel';
import { loadingDismiss, loadingFinish, loadingStage } from '../ui/loading';
import { installPanelToggle } from '../ui/panelToggle';
import { showCapabilityPage } from './capabilityPage';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const hud = document.getElementById('hud') as HTMLDivElement;
const panelContainer = document.getElementById('panel') as HTMLDivElement;
const panelToggle = document.getElementById('panel-toggle') as HTMLButtonElement;
const attribution = document.getElementById('attribution') as HTMLDivElement;

installPanelToggle(panelToggle);

/**
 * Startup presentation state. DEFAULT_PARAMETERS stays the scientific
 * baseline that presets and the validation record derive from; the app boots
 * into this presentation-layer overlay of it instead. Guards in the
 * ParameterStore still run on it like any other state.
 */
const STARTUP_PARAMETERS: Parameters = (() => {
  const p = cloneParameters(DEFAULT_PARAMETERS);
  p.camera.r = 60;
  p.camera.fovY = 1.0;
  p.disk.mdotKgPerS = 1e20; // log₁₀ Ṁ = 20.0
  p.disk.inhomogeneity = 1.0;
  p.bloomStrength = 0.1;
  p.timeScale = 100;
  return p;
})();

let renderer: Renderer | null = null;
let rafHandle = 0;
let panel: { dispose(): void } | null = null;

function canvasPixelSize(): [number, number] {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return [Math.max(1, Math.floor(canvas.clientWidth * dpr)), Math.max(1, Math.floor(canvas.clientHeight * dpr))];
}

function poseOf(params: Readonly<Parameters>): CameraPose {
  const c = params.camera;
  return { r: c.r, theta: c.theta, phi: c.phi, fovY: c.fovY, yaw: c.yaw, pitch: c.pitch };
}

function diskOf(params: Readonly<Parameters>): DiskParams {
  return { ...params.disk };
}

async function start(): Promise<void> {
  loadingStage('Initializing WebGPU', 0.15);
  let gpu;
  try {
    gpu = await acquireGpu(canvas, (info) => {
      // TDR or driver reset: tear down and rebuild the full GPU state.
      console.warn(`LensForge: GPU device lost (${info.reason ?? 'unknown'}): ${info.message} — recovering.`);
      cancelAnimationFrame(rafHandle);
      renderer = null;
      panel?.dispose();
      panel = null;
      void start();
    });
  } catch (err) {
    if (err instanceof WebGpuUnavailableError) {
      loadingDismiss();
      panelToggle.style.display = 'none';
      showCapabilityPage(canvas, err.message);
      return;
    }
    throw err;
  }

  // Startup micro-benchmark (M6): measured march workgroup size for this
  // device. Falls back to the default on any failure.
  loadingStage('Calibrating Compute Workgroups', 0.35);
  let marchWorkgroup: number | undefined;
  try {
    const bench = await benchmarkMarchWorkgroupSize(gpu);
    marchWorkgroup = bench.workgroupSize;
    console.info(
      `LensForge: march workgroup ${bench.workgroupSize} ` +
        `(${Object.entries(bench.timingsMs).map(([s, ms]) => `${s}: ${ms.toFixed(1)}ms`).join(', ')})`,
    );
  } catch (err) {
    console.warn('LensForge: workgroup benchmark failed — using default.', err);
  }

  loadingStage('Preparing Renderer', 0.55);
  const [w, h] = canvasPixelSize();
  canvas.width = w;
  canvas.height = h;
  const localRenderer = new Renderer(gpu, w, h, marchWorkgroup);
  renderer = localRenderer;

  const store = new ParameterStore(STARTUP_PARAMETERS);
  const clock = new SimClock();
  const exporter = new ImageExporter(localRenderer);
  let lastWarning: string | null = null;
  let exportStatus: string | null = null;

  // Store → renderer. The store has already classified and coerced.
  // During a tiled export the exporter owns the renderer (tile cameras, tile
  // buffer sizes); store changes are queued as a single re-apply afterwards.
  const applyPhysics = (params: Readonly<Parameters>): void => {
    localRenderer.setCamera(poseOf(params), params.spin, params.observer);
  };
  const applyAll = (params: Readonly<Parameters>): void => {
    localRenderer.tracePass.setDisk(diskOf(params));
    localRenderer.tracePass.setQuality(params.quality);
    applyPhysics(params);
    localRenderer.setExposure(Math.pow(2, params.exposureStops));
    localRenderer.setBloomStrength(params.bloomStrength);
  };
  store.subscribe((change, params) => {
    if (change.warning) lastWarning = change.warning;
    if (exporter.exporting) return; // re-applied wholesale after the export
    if (change.disk) localRenderer.tracePass.setDisk(diskOf(params));
    if (change.quality) localRenderer.tracePass.setQuality(params.quality);
    // Last, so a combined change (preset apply) lands with the new pose:
    // setDisk/setQuality re-derive uniforms from the pose they were given.
    if (change.kind === 'physics') applyPhysics(params);
    if (change.exposure) localRenderer.setExposure(Math.pow(2, params.exposureStops));
    if (change.bloom) localRenderer.setBloomStrength(params.bloomStrength);
  });

  const camera = new OrbitCamera(store);
  camera.attach(canvas);
  applyAll(store.get());

  panel = buildPanel(
    store,
    panelContainer,
    {
      shipped: SHIPPED_PRESETS.map((p) => ({ name: p.name, params: p.params })),
      save: (name, params) => serializePreset({ version: 2, name, params }),
      load: (json) => {
        const p = parsePreset(json);
        return { name: p.name, params: p.params };
      },
    },
    {
      screenshot: () => exporter.screenshot(),
      exportPng: async (width, height, onProgress) => {
        if (exporter.exporting) return;
        const params = store.get();
        try {
          await exporter.exportPng(
            { width, height, pose: poseOf(params), spin: params.spin, observer: params.observer },
            (done, total) => {
              exportStatus = `exporting ${width}×${height}: tile ${done}/${total}`;
              onProgress(done, total);
            },
          );
          exportStatus = null;
        } catch (err) {
          exportStatus = null;
          console.error('LensForge: export failed —', err);
        } finally {
          // The exporter restored preview/ladder state and canvas dimensions;
          // re-apply the store wholesale (camera, disk, quality, display).
          applyAll(store.get());
        }
      },
    },
  );

  // LUTs load async; the disk enables itself on arrival (physics restart).
  loadingStage('Loading Disk Radiance Tables', 0.7);
  const lutsReady = loadDiskLuts()
    .then((luts) => localRenderer.tracePass.setLuts(luts))
    .catch((err: unknown) => {
      console.warn('LensForge: disk LUTs unavailable — disk disabled.', err);
    });

  // Production sky (M7): cubemap + star catalog load async; until arrival the
  // sky renders black (dummies). In-app attribution is a licensing
  // requirement of the shipped data (IMPLEMENTATION.md M7, R18/R19).
  const skyReady = loadSkyAssets()
    .then(({ data, attribution: sources }) => {
      if (!renderer) return;
      localRenderer.tracePass.setSky(data);
      attribution.textContent = `Sky: ${sources.panorama.split(',')[0]} · Deep Star Maps 2020 (public domain) · Stars: Yale Bright Star Catalogue 5`;
    })
    .catch((err: unknown) => {
      console.warn('LensForge: sky assets unavailable — sky disabled.', err);
    });

  // The renderer runs behind the overlay meanwhile; once both asset loads
  // settle (success or fallback), fade into the already-converging image.
  // A plain timeout, not rAF: rAF never fires in a hidden tab and the
  // overlay must not outlive loading there.
  void lutsReady.then(() => loadingStage('Loading Physical Sky', 0.88));
  void Promise.allSettled([lutsReady, skyReady]).then(() => {
    window.setTimeout(() => loadingFinish(), 150);
  });

  new ResizeObserver(() => {
    const [rw, rh] = canvasPixelSize();
    canvas.width = rw;
    canvas.height = rh;
    localRenderer.resize(rw, rh);
    applyPhysics(store.get()); // aspect changed
  }).observe(canvas);

  // Decoupled loops (ARCHITECTURE.md §6): the sim loop advances t_sim and the
  // circular-orbit camera phase; the render loop refines or restarts.
  let lastNow = performance.now();
  let lastHudUpdate = 0;
  const frame = (now: number): void => {
    if (!renderer) return;
    const dtReal = Math.min(0.1, (now - lastNow) / 1000);
    lastNow = now;

    const params = store.get();
    const dtSim = clock.advance(dtReal, params.timeScale);
    if (!exporter.exporting) {
      if (params.observer === 'circular' && dtSim !== 0) {
        // Orbital phase advances at dφ/dt = Ω(r); a physics change per tick,
        // presented smoothly by the interaction preview. Static/ZAMO observers
        // with an axisymmetric disk are time-invariant: no restart (§6).
        store.update({ camera: { phi: params.camera.phi + angularVelocity(params.spin, params.camera.r) * dtSim } });
      }
      // M7 disk inhomogeneity advection: t_sim enters the trace uniforms; a
      // no-op (no restart) unless the pattern is enabled (PHYSICS.md §9.6).
      localRenderer.tracePass.setDiskTime(clock.tSim);

      renderer.renderFrame();
    }

    if (now - lastHudUpdate > 250) {
      const p = store.get();
      const marching = localRenderer.tracePass.getKnownMarching();
      const gov = localRenderer.tracePass.getGovernorState();
      const aa = localRenderer.tracePass.getRefinementState();
      hud.textContent =
        `LensForge ${APP_VERSION} — ${gpu.tier} tier — ${p.observer} — r=${p.camera.r.toFixed(1)}M — a=${p.spin.toFixed(3)}` +
        ` — t=${clock.tSim.toFixed(0)}M — budget ×${gov.workScale.toFixed(2)}` +
        `${marching > 0 ? ` — resolving ${marching} rays` : ` — AA ${aa.samples}/${aa.maxSamples}${aa.refining ? '…' : ''}`}` +
        `${exportStatus ? `\n⏳ ${exportStatus}` : ''}` +
        `${lastWarning ? `\n⚠ ${lastWarning}` : ''}\n${renderer.profiler.report()}`;
      lastHudUpdate = now;
    }
    rafHandle = requestAnimationFrame(frame);
  };
  rafHandle = requestAnimationFrame(frame);

  if (import.meta.env.DEV) {
    // Dev-only hooks: device-loss drill, rAF-independent rendering (hidden-tab
    // validation), and the V-O GPU-vs-oracle harness (src/validation).
    const setSpin = (a: number): void => void store.update({ spin: a });
    (window as unknown as Record<string, unknown>).__lensforge = {
      simulateDeviceLoss: () => gpu.device.destroy(),
      forceRecover: () => {
        cancelAnimationFrame(rafHandle);
        renderer = null;
        panel?.dispose();
        panel = null;
        void start();
      },
      profilerReport: () => localRenderer.profiler.report(),
      tracePass: localRenderer.tracePass,
      store,
      clock,
      resize: (rw: number, rh: number) => {
        canvas.width = rw;
        canvas.height = rh;
        localRenderer.resize(rw, rh);
        applyPhysics(store.get());
      },
      setCamera: (pose: Partial<Parameters['camera']>) => camera.set(pose),
      setSpin,
      getSpin: () => store.get().spin,
      setObserver: (o: Parameters['observer']) => store.update({ observer: o }),
      applyPreset: (name: string) => {
        const preset = SHIPPED_PRESETS.find((p) => p.name === name);
        if (!preset) throw new Error(`no preset named ${name}`);
        store.replace(preset.params);
      },
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
      runValidation: async (a = store.get().spin) => {
        const { runVOSuite } = await import('../validation/gpuOracleHarness');
        return runVOSuite(localRenderer.tracePass, a);
      },
      /** M5: V-O1 rerun through the boosted (circular-orbit) tetrad at r = 10. */
      runBoostedValidation: async (a = store.get().spin, r = 10) => {
        const { runVOBoostedSuite } = await import('../validation/gpuOracleHarness');
        return runVOBoostedSuite(localRenderer.tracePass, a, r);
      },
      measureShadow: async () => {
        const { measureShadowDiameter } = await import('../validation/gpuOracleHarness');
        store.update({ observer: 'zamo' });
        setSpin(0); // Synge prediction is the a = 0 closed form
        return measureShadowDiameter(localRenderer, camera);
      },
      /** V-I1: rendered silhouette vs Bardeen contour at the reference framing. */
      measureShadowContour: async (a = store.get().spin) => {
        const { measureShadowContour } = await import('../validation/gpuOracleHarness');
        store.update({ observer: 'zamo' });
        setSpin(a); // route the spin through the camera path before framing
        return measureShadowContour(localRenderer, camera, a);
      },
      setDisk: (p: Partial<DiskParams>) => {
        store.update({ disk: p });
      },
      getDisk: () => localRenderer.tracePass.getDisk(),
      readHdr: () => localRenderer.readHdr(),
      /** V-O2: GPU disk-hit (r, φ, g) vs float64 oracle. */
      runDiskValidation: async (a = store.get().spin) => {
        const { runVO2Suite } = await import('../validation/gpuOracleHarness');
        return runVO2Suite(localRenderer.tracePass, a);
      },
      /** V-I3 (oracle half): beaming asymmetry of the rendered disk vs oracle. */
      measureBeaming: async (a = store.get().spin) => {
        const { measureBeamingAsymmetry } = await import('../validation/gpuOracleHarness');
        store.update({ observer: 'zamo' });
        setSpin(a);
        return measureBeamingAsymmetry(localRenderer, camera, a);
      },
      /** M7: star-splat |det J|⁻¹ magnification vs oracle ray-bundle Jacobian. */
      measureStarMagnification: async (a = store.get().spin) => {
        const { measureStarMagnification } = await import('../validation/gpuOracleHarness');
        store.update({ observer: 'zamo' });
        setSpin(a);
        return measureStarMagnification(localRenderer, camera, a);
      },
      /** M7: V-I5 golden capture over all shipped presets (re-baseline). */
      captureGoldens: async () => {
        const { captureGolden } = await import('../validation/gpuOracleHarness');
        const entries = [];
        for (const preset of SHIPPED_PRESETS) {
          store.replace(preset.params);
          entries.push(await captureGolden(localRenderer, preset.name));
        }
        return entries;
      },
      /** M8 V-I2: disk-image boundary geometry at the Luminet framing. */
      runDiskEdgeValidation: async () => {
        const { measureDiskEdgeGeometry } = await import('../validation/gpuOracleHarness');
        const preset = SHIPPED_PRESETS.find((p) => p.name === 'Luminet 1979');
        if (!preset) throw new Error('Luminet 1979 preset missing');
        store.replace(preset.params);
        return measureDiskEdgeGeometry(localRenderer, camera, store.get().spin);
      },
      /** M8 V-O3: MAX_STEPS census over all shipped presets at convergence. */
      runStatusCensus: async () => {
        const { measureStatusCensus } = await import('../validation/gpuOracleHarness');
        const reports = [];
        for (const preset of SHIPPED_PRESETS) {
          store.replace(preset.params);
          reports.push(await measureStatusCensus(localRenderer, preset.name));
        }
        return reports;
      },
      /** M8 V-I5 re-baseline: converged displayed-image capture for a preset
       * (goldens may only change in a dedicated PR, VALIDATION.md §5). */
      captureGoldenImage: async (name: string) => {
        const { captureGoldenImage, presetSlug } = await import('../validation/gpuOracleHarness');
        const preset = SHIPPED_PRESETS.find((p) => p.name === name);
        if (!preset) throw new Error(`no preset named ${name}`);
        store.replace(preset.params);
        const img = await captureGoldenImage(localRenderer);
        return { slug: presetSlug(name), ...img };
      },
      /** M8 V-I5: SSIM + physical golden comparison, all shipped presets.
       * Renders at the baseline resolution, then restores the canvas. */
      runGoldenSuite: async () => {
        const { compareGolden, decodePngToLuma, presetSlug } = await import('../validation/gpuOracleHarness');
        const baselines = (await (await fetch('/tests/golden/baselines.json')).json()) as {
          resolution: string;
          entries: Array<{ name: string; meanLum: number; grid: number[] }>;
        };
        const match = /^(\d+)x(\d+)$/.exec(baselines.resolution);
        if (!match) throw new Error(`bad baseline resolution ${baselines.resolution}`);
        const [origW, origH] = [canvas.width, canvas.height];
        canvas.width = Number(match[1]);
        canvas.height = Number(match[2]);
        localRenderer.resize(canvas.width, canvas.height);
        try {
          const rows = [];
          for (const entry of baselines.entries) {
            const preset = SHIPPED_PRESETS.find((p) => p.name === entry.name);
            if (!preset) throw new Error(`baseline ${entry.name} has no shipped preset`);
            const res = await fetch(`/tests/golden/${presetSlug(entry.name)}.png`);
            if (!res.ok) throw new Error(`golden PNG missing for ${entry.name}`);
            const luma = await decodePngToLuma(await res.blob());
            store.replace(preset.params);
            rows.push(await compareGolden(localRenderer, entry, luma));
          }
          return { rows, pass: rows.every((r) => r.pass) };
        } finally {
          canvas.width = origW;
          canvas.height = origH;
          localRenderer.resize(origW, origH);
          applyPhysics(store.get());
        }
      },
      exporter,
      /** Present transform → rgba8 bytes (visual validation captures). */
      presentBytes: () => localRenderer.presentToBytes(),
      exportPng: (width: number, height: number, samples = 8) => {
        const p = store.get();
        return exporter.exportPng(
          { width, height, pose: poseOf(p), spin: p.spin, observer: p.observer, samples },
          (done, total) => console.info(`export tile ${done}/${total}`),
        );
      },
    };
  }
}

void start();
