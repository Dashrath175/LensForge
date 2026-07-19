/**
 * ParameterStore (ARCHITECTURE.md §1, §6): the single source of truth for
 * render state. UI and camera controller write here and nowhere else; the
 * app subscribes and drives the renderer. Every update is classified so the
 * subscriber can decide restart-vs-continue:
 *
 *   'physics'  — anything entering the trace uniforms (spin, observer,
 *                camera, disk, quality): accumulation must restart.
 *   'display'  — present-pass state (exposure): no restart.
 *   'time'     — timeScale: nothing restarts; the sim loop reads it.
 *   'none'     — no observable change (e.g. camera drag mode).
 *
 * Observer validity guards (IMPLEMENTATION.md M5) are enforced here by
 * coercion, so downstream physics never sees an invalid observer:
 *   static   — must sit outside the ergosphere, else falls back to ZAMO;
 *   circular — equatorial by definition (θ forced to π/2) and restricted to
 *              stable orbits (r clamped to ≥ r_isco).
 */
import {
  cloneParameters,
  DEFAULT_PARAMETERS,
  FOV_MAX,
  FOV_MIN,
  PITCH_LIMIT,
  R_MAX,
  R_MIN,
  SPIN_MAX,
  THETA_LIMIT,
  type CameraState,
  type DiskSettings,
  type Parameters,
} from '../core/parameters';
import { ergosphereRadius } from '../physics/metric';
import { iscoRadius } from '../physics/orbits';

export type ChangeKind = 'physics' | 'display' | 'time' | 'none';

/** Which top-level groups changed — lets the subscriber apply narrowly. */
export interface ChangeSet {
  kind: ChangeKind;
  spin: boolean;
  observer: boolean;
  camera: boolean;
  disk: boolean;
  quality: boolean;
  exposure: boolean;
  bloom: boolean;
  timeScale: boolean;
  /** Set when a guard coerced the request (observer fallback, r clamp). */
  warning: string | null;
}

export interface ParameterPatch {
  spin?: number;
  observer?: Parameters['observer'];
  camera?: Partial<CameraState>;
  disk?: Partial<DiskSettings>;
  quality?: Parameters['quality'];
  exposureStops?: number;
  bloomStrength?: number;
  timeScale?: number;
}

export type StoreListener = (change: ChangeSet, params: Readonly<Parameters>) => void;

const clamp = (x: number, lo: number, hi: number): number => Math.min(Math.max(x, lo), hi);

export class ParameterStore {
  private params: Parameters;
  private readonly listeners = new Set<StoreListener>();

  constructor(initial: Parameters = DEFAULT_PARAMETERS) {
    this.params = this.coerce(cloneParameters(initial)).params;
  }

  get(): Readonly<Parameters> {
    return this.params;
  }

  /** A detached deep copy (preset capture). */
  snapshot(): Parameters {
    return cloneParameters(this.params);
  }

  subscribe(listener: StoreListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  update(patch: ParameterPatch): ChangeSet {
    const next = cloneParameters(this.params);
    if (patch.spin !== undefined) next.spin = patch.spin;
    if (patch.observer !== undefined) next.observer = patch.observer;
    if (patch.camera) Object.assign(next.camera, patch.camera);
    if (patch.disk) Object.assign(next.disk, patch.disk);
    if (patch.quality !== undefined) next.quality = patch.quality;
    if (patch.exposureStops !== undefined) next.exposureStops = patch.exposureStops;
    if (patch.bloomStrength !== undefined) next.bloomStrength = patch.bloomStrength;
    if (patch.timeScale !== undefined) next.timeScale = patch.timeScale;
    return this.commit(next);
  }

  /** Replace the whole state (preset apply); guards still run. */
  replace(params: Parameters): ChangeSet {
    return this.commit(cloneParameters(params));
  }

  private commit(candidate: Parameters): ChangeSet {
    const { params: next, warning } = this.coerce(candidate);
    const prev = this.params;

    const cameraChanged = (Object.keys(next.camera) as Array<keyof CameraState>).some(
      (k) => next.camera[k] !== prev.camera[k],
    );
    const cameraModeOnly =
      cameraChanged &&
      (Object.keys(next.camera) as Array<keyof CameraState>).every(
        (k) => k === 'mode' || next.camera[k] === prev.camera[k],
      );
    const diskChanged = (Object.keys(next.disk) as Array<keyof DiskSettings>).some(
      (k) => next.disk[k] !== prev.disk[k],
    );

    const change: ChangeSet = {
      kind: 'none',
      spin: next.spin !== prev.spin,
      observer: next.observer !== prev.observer,
      camera: cameraChanged && !cameraModeOnly,
      disk: diskChanged,
      quality: next.quality !== prev.quality,
      exposure: next.exposureStops !== prev.exposureStops,
      bloom: next.bloomStrength !== prev.bloomStrength,
      timeScale: next.timeScale !== prev.timeScale,
      warning,
    };
    if (change.spin || change.observer || change.camera || change.disk || change.quality) {
      change.kind = 'physics';
    } else if (change.exposure || change.bloom) {
      change.kind = 'display';
    } else if (change.timeScale) {
      change.kind = 'time';
    } else if (cameraModeOnly) {
      change.kind = 'none';
    }

    this.params = next;
    if (change.kind !== 'none' || cameraModeOnly || warning) {
      for (const l of this.listeners) l(change, this.params);
    }
    return change;
  }

  /** Range clamps + observer validity coercion. Pure; returns the warning. */
  private coerce(p: Parameters): { params: Parameters; warning: string | null } {
    let warning: string | null = null;
    p.spin = clamp(p.spin, 0, SPIN_MAX);
    const c = p.camera;
    c.r = clamp(c.r, R_MIN, R_MAX);
    c.theta = clamp(c.theta, THETA_LIMIT, Math.PI - THETA_LIMIT);
    c.fovY = clamp(c.fovY, FOV_MIN, FOV_MAX);
    c.pitch = clamp(c.pitch, -PITCH_LIMIT, PITCH_LIMIT);
    p.disk.rOuter = clamp(p.disk.rOuter, iscoRadius(p.spin) + 0.5, 100);
    p.disk.inhomogeneity = clamp(p.disk.inhomogeneity, 0, 1);
    p.bloomStrength = clamp(p.bloomStrength, 0, 0.5);

    if (p.observer === 'circular') {
      // Equatorial by definition; stable-orbit branch only (PHYSICS.md §6).
      c.theta = Math.PI / 2;
      const rIsco = iscoRadius(p.spin);
      if (c.r < rIsco) {
        c.r = rIsco;
        warning = `circular orbit clamped to ISCO (r = ${rIsco.toFixed(3)}M)`;
      }
    } else if (p.observer === 'static') {
      const rErgo = ergosphereRadius(p.spin, c.theta);
      if (c.r <= rErgo * (1 + 1e-6)) {
        p.observer = 'zamo';
        warning = `static observer invalid inside ergosphere (r ≤ ${rErgo.toFixed(3)}M) — using ZAMO`;
      }
    }
    return { params: p, warning };
  }
}
