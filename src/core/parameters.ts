/**
 * Canonical parameter vocabulary (ARCHITECTURE.md §6): the full render state
 * shared by ParameterStore (src/sim), presets (src/scene), and the UI
 * (src/ui). Lives in core so no module boundary is crossed to name a
 * parameter. Semantics and guards live in src/sim/parameterStore.ts.
 */

/** Observer 4-velocity family (PHYSICS.md §5). */
export type ObserverType = 'zamo' | 'static' | 'circular';

/** Pointer-drag interpretation: orbit the hole vs free-look from a fixed point. */
export type CameraMode = 'orbit' | 'look';

export type QualityTierName = 'low' | 'medium' | 'high';

export interface CameraState {
  /** Boyer-Lindquist radius of the observer, GM/c². */
  r: number;
  /** BL colatitude; clamped off the exact pole (tetrad φ-leg degenerates). */
  theta: number;
  phi: number;
  /** Free-look offsets, radians: rotation of the local view triad. Zero means
   * the view axis points at the hole (the orbit-camera convention). */
  yaw: number;
  pitch: number;
  fovY: number;
  mode: CameraMode;
}

export interface DiskSettings {
  enabled: boolean;
  /** Outer edge in GM/c² (inner edge is always the ISCO). */
  rOuter: number;
  massKg: number;
  mdotKgPerS: number;
  /** PHYSICS.md §8: 'physical' = true T_eff; 'scaled' = global T rescale. */
  colorMode: 'physical' | 'scaled';
  /** M7 statistical inhomogeneity amplitude (PHYSICS.md §9.6), 0 = off:
   * relative flux contrast of a mean-1 brightness field advected at the
   * exact Ω(r). A labeled statistical model, not GRMHD. */
  inhomogeneity: number;
}

export interface Parameters {
  spin: number;
  observer: ObserverType;
  camera: CameraState;
  disk: DiskSettings;
  quality: QualityTierName;
  /** Display-only exposure in stops (present pass; no accumulation restart). */
  exposureStops: number;
  /** M7 optional bloom mix (camera model, documented artistic layer): the
   * presented image is mix(hdr, blurred hdr, bloomStrength) — an
   * energy-conserving redistribution. 0 = off (default). Display-only. */
  bloomStrength: number;
  /** Simulation seconds of coordinate time (GM/c³ units) per real second. */
  timeScale: number;
}

/** Thorne limit (IMPLEMENTATION.md M3); also keeps Δ cancellation manageable in f32. */
export const SPIN_MAX = 0.998;
/** Camera position must stay off the exact pole (observer tetrad degenerates). */
export const THETA_LIMIT = 1e-3;
export const R_MIN = 4;
export const R_MAX = 5000;
export const FOV_MIN = 0.002;
export const FOV_MAX = 2.0;
/** Free-look pitch limit: keep the pole flip unreachable. */
export const PITCH_LIMIT = 1.45;

export const DEFAULT_PARAMETERS: Parameters = {
  spin: 0,
  observer: 'zamo',
  camera: { r: 50, theta: Math.PI / 2, phi: 0, yaw: 0, pitch: 0, fovY: 0.6, mode: 'orbit' },
  disk: {
    enabled: true,
    rOuter: 20,
    massKg: 1.989e37, // 1e7 solar masses
    mdotKgPerS: 1.4e21, // ~0.1 Eddington at 10% efficiency for that mass
    colorMode: 'scaled',
    inhomogeneity: 0,
  },
  quality: 'high',
  exposureStops: 0,
  bloomStrength: 0,
  timeScale: 0,
};

/** Deep copy (parameters are plain data). */
export function cloneParameters(p: Parameters): Parameters {
  return {
    ...p,
    camera: { ...p.camera },
    disk: { ...p.disk },
  };
}
