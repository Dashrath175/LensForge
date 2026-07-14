/**
 * Minimal BH-centered orbit camera for M2 (expanded into the full observer
 * system in M5): drag orbits the viewpoint over the (θ, φ) sphere at radius r,
 * wheel dollies r. The view axis always points at the hole — orientation is
 * carried entirely by the observer tetrad (PHYSICS.md §5).
 */
import type { CameraPose } from '../renderer/tracePass';

const DRAG_RADIANS_PER_PIXEL = 0.004;
const WHEEL_DOLLY_RATE = 0.0012;
const THETA_LIMIT = 0.15; // keep clear of the BL polar axis until M3's axis handling
const R_MIN = 4;
const R_MAX = 5000;

export const DEFAULT_POSE: CameraPose = {
  r: 50,
  theta: Math.PI / 2,
  phi: 0,
  fovY: 0.6,
};

export class OrbitCamera {
  readonly pose: CameraPose = { ...DEFAULT_POSE };
  private readonly onChange: () => void;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  constructor(onChange: () => void) {
    this.onChange = onChange;
  }

  set(partial: Partial<CameraPose>): void {
    Object.assign(this.pose, partial);
    this.clamp();
    this.onChange();
  }

  private clamp(): void {
    this.pose.theta = Math.min(Math.max(this.pose.theta, THETA_LIMIT), Math.PI - THETA_LIMIT);
    this.pose.r = Math.min(Math.max(this.pose.r, R_MIN), R_MAX);
    this.pose.fovY = Math.min(Math.max(this.pose.fovY, 0.02), 2.0);
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
      this.pose.phi -= (e.clientX - this.lastX) * DRAG_RADIANS_PER_PIXEL;
      this.pose.theta -= (e.clientY - this.lastY) * DRAG_RADIANS_PER_PIXEL;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.clamp();
      this.onChange();
    });
    canvas.addEventListener('pointerup', () => {
      this.dragging = false;
    });
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.pose.r *= Math.exp(e.deltaY * WHEEL_DOLLY_RATE);
        this.clamp();
        this.onChange();
      },
      { passive: false },
    );
  }
}
