/**
 * Tweakpane control panel (ARCHITECTURE.md §2): bindings write to the
 * ParameterStore and NOTHING else — the app subscribes to the store and
 * drives the renderer. The panel also subscribes, so external writes (camera
 * drags, preset apply, guard coercions) refresh the controls.
 */
import { Pane } from 'tweakpane';
import './theme.css';
import { SPIN_MAX, type Parameters } from '../core/parameters';
import type { ParameterStore } from '../sim/parameterStore';
import { showAbout } from './about';

export interface PanelHandle {
  dispose(): void;
}

/** Preset IO, injected by the app (ui depends on sim + core only,
 * ARCHITECTURE.md §7 — preset serialization lives in src/scene). */
export interface PresetIO {
  shipped: ReadonlyArray<{ name: string; params: Parameters }>;
  save(name: string, params: Parameters): string;
  load(json: string): { name: string; params: Parameters };
}

/** Export IO (M7), injected by the app (the exporter lives in src/renderer). */
export interface ExportIO {
  /** PNG of the currently presented frame at canvas resolution. */
  screenshot(): Promise<void>;
  /** Tiled high-resolution PNG export (up to 8K). */
  exportPng(width: number, height: number, onProgress: (done: number, total: number) => void): Promise<void>;
}

const EXPORT_RESOLUTIONS: Record<string, [number, number]> = {
  '1920 × 1080': [1920, 1080],
  '2560 × 1440': [2560, 1440],
  '3840 × 2160 (4K)': [3840, 2160],
  '7680 × 4320 (8K)': [7680, 4320],
};

export function buildPanel(
  store: ParameterStore,
  container: HTMLElement,
  presetIo: PresetIO,
  exportIo: ExportIO,
): PanelHandle {
  const pane = new Pane({ container, title: 'LensForge' });

  // Working copy the pane binds to; store subscription keeps it in sync.
  const ui = flatten(store.get());
  let applying = false;
  const push = (fn: () => void): void => {
    if (applying) return;
    fn();
  };

  const spacetime = pane.addFolder({ title: 'Spacetime' });
  spacetime
    .addBinding(ui, 'spin', { min: 0, max: SPIN_MAX, step: 0.001 })
    .on('change', (ev) => push(() => store.update({ spin: ev.value })));
  spacetime
    .addBinding(ui, 'observer', {
      options: { ZAMO: 'zamo', Static: 'static', 'Circular orbit': 'circular' },
    })
    .on('change', (ev) => push(() => store.update({ observer: ev.value as Parameters['observer'] })));
  spacetime
    .addBinding(ui, 'quality', { options: { High: 'high', Medium: 'medium', Low: 'low' } })
    .on('change', (ev) => push(() => store.update({ quality: ev.value as Parameters['quality'] })));

  const camera = pane.addFolder({ title: 'Camera' });
  camera
    .addBinding(ui, 'mode', { options: { Orbit: 'orbit', 'Free look': 'look' } })
    .on('change', (ev) => push(() => store.update({ camera: { mode: ev.value as 'orbit' | 'look' } })));
  camera
    .addBinding(ui, 'distance', { min: 4, max: 1000, step: 0.5, label: 'distance (M)' })
    .on('change', (ev) => push(() => store.update({ camera: { r: ev.value } })));
  camera
    .addBinding(ui, 'fovY', { min: 0.002, max: 2.0, step: 0.002, label: 'fov (rad)' })
    .on('change', (ev) => push(() => store.update({ camera: { fovY: ev.value } })));
  camera.addButton({ title: 'Reset view' }).on('click', () =>
    store.update({ camera: { theta: Math.PI / 2, phi: 0, yaw: 0, pitch: 0 } }),
  );

  const disk = pane.addFolder({ title: 'Disk' });
  disk
    .addBinding(ui, 'diskEnabled', { label: 'enabled' })
    .on('change', (ev) => push(() => store.update({ disk: { enabled: ev.value } })));
  disk
    .addBinding(ui, 'colorMode', { options: { 'Temp-scaled': 'scaled', Physical: 'physical' }, label: 'colors' })
    .on('change', (ev) => push(() => store.update({ disk: { colorMode: ev.value as 'scaled' | 'physical' } })));
  disk
    .addBinding(ui, 'log10Mdot', { min: 19, max: 23, step: 0.05, label: 'log₁₀ Ṁ' })
    .on('change', (ev) => push(() => store.update({ disk: { mdotKgPerS: Math.pow(10, ev.value) } })));
  disk
    .addBinding(ui, 'rOuter', { min: 8, max: 40, step: 1, label: 'r_out (M)' })
    .on('change', (ev) => push(() => store.update({ disk: { rOuter: ev.value } })));
  // M7 statistical inhomogeneity (PHYSICS.md §9.6) — explicitly labeled a
  // statistical model: transport is the exact Ω(r), the texture is not GRMHD.
  disk
    .addBinding(ui, 'inhomogeneity', { min: 0, max: 1, step: 0.01, label: 'turbulence\n(stat. model)' })
    .on('change', (ev) => push(() => store.update({ disk: { inhomogeneity: ev.value } })));

  const display = pane.addFolder({ title: 'Display & Time' });
  display
    .addBinding(ui, 'exposureStops', { min: -6, max: 6, step: 0.1, label: 'exposure' })
    .on('change', (ev) => push(() => store.update({ exposureStops: ev.value })));
  // M7 optional bloom (documented camera model; energy-conserving mix).
  display
    .addBinding(ui, 'bloomStrength', { min: 0, max: 0.5, step: 0.01, label: 'bloom' })
    .on('change', (ev) => push(() => store.update({ bloomStrength: ev.value })));
  display
    .addBinding(ui, 'timeScale', { min: 0, max: 500, step: 1, label: 'time (M/s)' })
    .on('change', (ev) => push(() => store.update({ timeScale: ev.value })));

  // M7 export: PNG screenshot + tiled high-res export via the chunk scheduler.
  const exportFolder = pane.addFolder({ title: 'Export' });
  const exportState = { resolution: '3840 × 2160 (4K)' };
  const resOptions: Record<string, string> = {};
  for (const k of Object.keys(EXPORT_RESOLUTIONS)) resOptions[k] = k;
  exportFolder.addBinding(exportState, 'resolution', { options: resOptions });
  exportFolder.addButton({ title: 'Screenshot PNG' }).on('click', () => {
    void exportIo.screenshot();
  });
  const exportButton = exportFolder.addButton({ title: 'Export PNG (tiled)' });
  let exporting = false;
  exportButton.on('click', () => {
    if (exporting) return;
    exporting = true;
    const [w, h] = EXPORT_RESOLUTIONS[exportState.resolution]!;
    void exportIo
      .exportPng(w, h, (done, total) => {
        exportButton.title = `Exporting… ${done}/${total}`;
      })
      .finally(() => {
        exporting = false;
        exportButton.title = 'Export PNG (tiled)';
      });
  });

  const presets = pane.addFolder({ title: 'Presets' });
  const presetState = { name: presetIo.shipped[0]?.name ?? '' };
  const options: Record<string, string> = {};
  for (const p of presetIo.shipped) options[p.name] = p.name;
  presets.addBinding(presetState, 'name', { options, label: 'preset' });
  presets.addButton({ title: 'Apply preset' }).on('click', () => {
    const preset = presetIo.shipped.find((p) => p.name === presetState.name);
    if (preset) store.replace(preset.params);
  });
  presets.addButton({ title: 'Save JSON' }).on('click', () => {
    const json = presetIo.save(presetState.name, store.snapshot());
    const blob = new Blob([json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `lensforge-${presetState.name.toLowerCase().replace(/\s+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
  presets.addButton({ title: 'Load JSON…' }).on('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (!file) return;
      void file.text().then((text) => {
        try {
          store.replace(presetIo.load(text).params);
        } catch (err) {
          console.warn('LensForge: preset rejected —', err);
        }
      });
    });
    input.click();
  });

  // M8: the accuracy boundaries of the simulation, surfaced in-product.
  pane.addButton({ title: 'About & accuracy' }).on('click', () => showAbout());

  const unsubscribe = store.subscribe((_change, params) => {
    applying = true;
    Object.assign(ui, flatten(params));
    pane.refresh();
    applying = false;
  });

  return {
    dispose(): void {
      unsubscribe();
      pane.dispose();
    },
  };
}

/** Store state → flat pane-bindable fields. */
function flatten(p: Readonly<Parameters>): {
  spin: number;
  observer: string;
  quality: string;
  mode: string;
  distance: number;
  fovY: number;
  diskEnabled: boolean;
  colorMode: string;
  log10Mdot: number;
  rOuter: number;
  inhomogeneity: number;
  exposureStops: number;
  bloomStrength: number;
  timeScale: number;
} {
  return {
    spin: p.spin,
    observer: p.observer,
    quality: p.quality,
    mode: p.camera.mode,
    distance: p.camera.r,
    fovY: p.camera.fovY,
    diskEnabled: p.disk.enabled,
    colorMode: p.disk.colorMode,
    log10Mdot: Math.log10(p.disk.mdotKgPerS),
    rOuter: p.disk.rOuter,
    inhomogeneity: p.disk.inhomogeneity,
    exposureStops: p.exposureStops,
    bloomStrength: p.bloomStrength,
    timeScale: p.timeScale,
  };
}
