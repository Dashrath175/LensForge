/**
 * Preset schema + canonical serialization (IMPLEMENTATION.md M5).
 *
 * A preset is the FULL parameter + camera state — applying one reproduces a
 * framing exactly. Serialization is canonical (fixed key order, fixed
 * indentation), so save → load → save is byte-identical; the round-trip is a
 * unit test (tests/unit/scene/presets.test.ts).
 *
 * The V-I reference framings live HERE and are imported by the validation
 * harness (src/validation must not ship in the production bundle, so the
 * dependency points this way): presets and validation share one source of
 * truth, per the M5 implementation note.
 */
import {
  DEFAULT_PARAMETERS,
  cloneParameters,
  type CameraState,
  type Parameters,
} from '../core/parameters';
import { criticalImpactParameter } from '../physics/shadow';

/** V-I1 reference framing (VALIDATION.md §3): equatorial ZAMO at r = 1000M. */
export const VI1_FRAMING = { r: 1000, theta: Math.PI / 2, phi: 0, fovY: 0.03 } as const;
/** V-I3 reference framing: near-edge-on ZAMO at r = 60M (beaming-dominated). */
export const VI3_FRAMING = { r: 60, theta: 1.396, phi: 0, fovY: 0.7 } as const;

/** Schema version 2 (M7): adds disk.inhomogeneity and bloomStrength. Version
 * 1 presets are accepted on load with both set to 0 — exactly the pre-M7
 * behavior, so old presets reproduce their framings unchanged. */
export const PRESET_VERSION = 2;

export interface Preset {
  version: typeof PRESET_VERSION;
  name: string;
  params: Parameters;
}

/** Canonical form: explicit field order, 2-space indent. Unknown fields are
 * dropped; missing fields are a parse error (a preset is a full state). */
export function serializePreset(preset: Preset): string {
  const { params: p } = preset;
  const canonical = {
    version: preset.version,
    name: preset.name,
    params: {
      spin: p.spin,
      observer: p.observer,
      camera: {
        r: p.camera.r,
        theta: p.camera.theta,
        phi: p.camera.phi,
        yaw: p.camera.yaw,
        pitch: p.camera.pitch,
        fovY: p.camera.fovY,
        mode: p.camera.mode,
      },
      disk: {
        enabled: p.disk.enabled,
        rOuter: p.disk.rOuter,
        massKg: p.disk.massKg,
        mdotKgPerS: p.disk.mdotKgPerS,
        colorMode: p.disk.colorMode,
        inhomogeneity: p.disk.inhomogeneity,
      },
      quality: p.quality,
      exposureStops: p.exposureStops,
      bloomStrength: p.bloomStrength,
      timeScale: p.timeScale,
    },
  };
  return JSON.stringify(canonical, null, 2);
}

function expectNumber(v: unknown, path: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`preset: ${path} must be a finite number`);
  return v;
}

function expectEnum<T extends string>(v: unknown, allowed: readonly T[], path: string): T {
  if (typeof v !== 'string' || !(allowed as readonly string[]).includes(v)) {
    throw new Error(`preset: ${path} must be one of ${allowed.join(', ')}`);
  }
  return v as T;
}

export function parsePreset(json: string): Preset {
  const raw = JSON.parse(json) as Record<string, unknown>;
  const version = raw['version'];
  if (version !== 1 && version !== PRESET_VERSION) throw new Error('preset: unsupported version');
  if (typeof raw['name'] !== 'string' || raw['name'].length === 0) throw new Error('preset: missing name');
  const p = raw['params'] as Record<string, unknown> | undefined;
  if (!p || typeof p !== 'object') throw new Error('preset: missing params');
  const cam = p['camera'] as Record<string, unknown> | undefined;
  const disk = p['disk'] as Record<string, unknown> | undefined;
  if (!cam || !disk) throw new Error('preset: missing camera/disk');
  // Version-1 presets predate the M7 fields; 0 reproduces their behavior.
  if (version === 1) {
    if (disk['inhomogeneity'] === undefined) disk['inhomogeneity'] = 0;
    if (p['bloomStrength'] === undefined) p['bloomStrength'] = 0;
  }

  const params: Parameters = {
    spin: expectNumber(p['spin'], 'params.spin'),
    observer: expectEnum(p['observer'], ['zamo', 'static', 'circular'] as const, 'params.observer'),
    camera: {
      r: expectNumber(cam['r'], 'camera.r'),
      theta: expectNumber(cam['theta'], 'camera.theta'),
      phi: expectNumber(cam['phi'], 'camera.phi'),
      yaw: expectNumber(cam['yaw'], 'camera.yaw'),
      pitch: expectNumber(cam['pitch'], 'camera.pitch'),
      fovY: expectNumber(cam['fovY'], 'camera.fovY'),
      mode: expectEnum(cam['mode'], ['orbit', 'look'] as const, 'camera.mode'),
    },
    disk: {
      enabled: Boolean(disk['enabled']),
      rOuter: expectNumber(disk['rOuter'], 'disk.rOuter'),
      massKg: expectNumber(disk['massKg'], 'disk.massKg'),
      mdotKgPerS: expectNumber(disk['mdotKgPerS'], 'disk.mdotKgPerS'),
      colorMode: expectEnum(disk['colorMode'], ['physical', 'scaled'] as const, 'disk.colorMode'),
      inhomogeneity: expectNumber(disk['inhomogeneity'], 'disk.inhomogeneity'),
    },
    quality: expectEnum(p['quality'], ['low', 'medium', 'high'] as const, 'params.quality'),
    exposureStops: expectNumber(p['exposureStops'], 'params.exposureStops'),
    bloomStrength: expectNumber(p['bloomStrength'], 'params.bloomStrength'),
    timeScale: expectNumber(p['timeScale'], 'params.timeScale'),
  };
  return { version: PRESET_VERSION, name: raw['name'], params };
}

function withDefaults(over: {
  spin?: number;
  observer?: Parameters['observer'];
  camera?: Partial<CameraState>;
  disk?: Partial<Parameters['disk']>;
  exposureStops?: number;
  timeScale?: number;
}): Parameters {
  const p = cloneParameters(DEFAULT_PARAMETERS);
  if (over.spin !== undefined) p.spin = over.spin;
  if (over.observer !== undefined) p.observer = over.observer;
  Object.assign(p.camera, over.camera);
  Object.assign(p.disk, over.disk);
  if (over.exposureStops !== undefined) p.exposureStops = over.exposureStops;
  if (over.timeScale !== undefined) p.timeScale = over.timeScale;
  return p;
}

/**
 * Shipped presets (IMPLEMENTATION.md M5). The two literature framings reuse
 * the V-I reference constants verbatim; "Photon-ring grazing" centers the
 * frame on the prograde critical curve via the exact Bardeen impact
 * parameter (small-angle: yaw ≈ b_c/r at r = 1000M).
 */
export const SHIPPED_PRESETS: readonly Preset[] = [
  {
    version: PRESET_VERSION,
    name: 'Luminet 1979',
    // Luminet's bare-photograph geometry: Schwarzschild, disk to 20M, viewed
    // ~10° above the plane (V-I3 framing is that geometry at r = 60M).
    params: withDefaults({ spin: 0, camera: { ...VI3_FRAMING }, disk: { enabled: true, rOuter: 20 } }),
  },
  {
    version: PRESET_VERSION,
    name: 'DNGR Gargantua',
    // Near-extremal spin at the Thorne cap, near-edge-on, temperature-scaled
    // colors — the Interstellar viewing geometry [R5].
    params: withDefaults({
      spin: 0.998,
      camera: { r: 40, theta: 1.48, phi: 0, fovY: 0.8 },
      disk: { enabled: true, rOuter: 20, colorMode: 'scaled' },
    }),
  },
  {
    version: PRESET_VERSION,
    name: 'Face-on',
    params: withDefaults({
      spin: 0.9,
      camera: { r: 80, theta: 0.05, phi: 0, fovY: 0.5 },
      disk: { enabled: true, rOuter: 20 },
    }),
  },
  {
    version: PRESET_VERSION,
    name: 'Photon-ring grazing',
    params: withDefaults({
      spin: 0.9,
      camera: {
        ...VI1_FRAMING,
        fovY: 0.004,
        yaw: criticalImpactParameter(0.9, true) / VI1_FRAMING.r,
      },
      disk: { enabled: false },
    }),
  },
];
