/**
 * Startup loading overlay controller (presentation only). The overlay itself
 * is static markup in index.html so it renders before any module loads; this
 * module advances its stage text/progress and fades it out over the first
 * presented frames. Everything here is a no-op once the overlay is gone —
 * device-loss recovery re-runs the bootstrap without re-showing it.
 */
import { APP_VERSION } from '../core/version';

const overlay = document.getElementById('loading');
const fill = document.getElementById('loading-fill');
const stageEl = document.getElementById('loading-stage');
const versionEl = document.getElementById('loading-version');
if (versionEl) versionEl.textContent = `v${APP_VERSION}`;

let finished = false;
let stageToken = 0;

/** Advance the stage label (crossfade) and the progress fill. */
export function loadingStage(label: string, fraction: number): void {
  if (finished || !overlay) return;
  if (fill) fill.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
  if (!stageEl || stageEl.textContent === label) return;
  const token = ++stageToken;
  stageEl.style.opacity = '0';
  window.setTimeout(() => {
    if (finished || token !== stageToken) return;
    stageEl.textContent = label;
    stageEl.style.opacity = '1';
  }, 150);
}

/** Complete the bar, fade the overlay out, and remove it from the DOM. */
export function loadingFinish(): void {
  if (finished || !overlay) return;
  finished = true;
  if (fill) fill.style.width = '100%';
  window.setTimeout(() => overlay.classList.add('done'), 250);
  window.setTimeout(() => overlay.remove(), 1000);
}

/** Remove the overlay immediately (WebGPU-unavailable capability page). */
export function loadingDismiss(): void {
  finished = true;
  overlay?.remove();
}
