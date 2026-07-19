/**
 * Camera controller (M5): pointer/wheel input → ParameterStore, nothing else
 * (ARCHITECTURE.md §1 — the store is the single source of truth; the app
 * subscribes and drives the renderer).
 *
 * Two drag modes (Parameters.camera.mode):
 *   orbit — drag moves the viewpoint over the (θ, φ) sphere; the view axis
 *           points at the hole. Wheel dollies r.
 *   look  — drag rotates the local view triad (yaw/pitch) from a fixed
 *           position; orientation is carried by a CPU-side spatial rotation
 *           of the observer tetrad (PHYSICS.md §5), so aberration and all
 *           physics are untouched. Wheel still dollies r.
 */
import type { CameraState } from '../core/parameters';
import type { ParameterStore } from '../sim/parameterStore';
import type { CameraPose } from '../renderer/tracePass';

const DRAG_RADIANS_PER_PIXEL = 0.004;
const WHEEL_DOLLY_RATE = 0.0012;

export class OrbitCamera {
  private readonly store: ParameterStore;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  constructor(store: ParameterStore) {
    this.store = store;
  }

  /** Current pose in renderer terms (the store owns the state). */
  get pose(): CameraPose {
    const c = this.store.get().camera;
    return { r: c.r, theta: c.theta, phi: c.phi, fovY: c.fovY, yaw: c.yaw, pitch: c.pitch };
  }

  /** Merge a partial camera state (validation framings, presets, dev hooks).
   * Framings that omit yaw/pitch reset them — a framing is a full pose. */
  set(partial: Partial<CameraState>): void {
    this.store.update({ camera: { yaw: 0, pitch: 0, ...partial } });
  }

  attach(canvas: HTMLCanvasElement): void {
    canvas.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const dx = (e.clientX - this.lastX) * DRAG_RADIANS_PER_PIXEL;
      const dy = (e.clientY - this.lastY) * DRAG_RADIANS_PER_PIXEL;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      const c = this.store.get().camera;
      // Horizontal deltas are negated relative to M5: the M7 present pass
      // mirrors the displayed image in screen x (arriving-photon parity,
      // shaders/post/present.wgsl), and the input mapping mirrors with it so
      // on-screen drag behavior is unchanged.
      if (c.mode === 'look') {
        this.store.update({ camera: { yaw: c.yaw - dx, pitch: c.pitch + dy } });
      } else {
        this.store.update({ camera: { phi: c.phi + dx, theta: c.theta - dy } });
      }
    });
    canvas.addEventListener('pointerup', () => {
      this.dragging = false;
    });
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const c = this.store.get().camera;
        this.store.update({ camera: { r: c.r * Math.exp(e.deltaY * WHEEL_DOLLY_RATE) } });
      },
      { passive: false },
    );
  }
}
