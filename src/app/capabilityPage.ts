/** WebGPU-unavailable fallback page (D1: capability page instead of a WebGL2 path). */
export function showCapabilityPage(canvas: HTMLCanvasElement, detail: string): void {
  canvas.style.display = 'none';
  const hud = document.getElementById('hud');
  if (hud) hud.style.display = 'none';
  const el = document.getElementById('fallback');
  if (!el) return;
  el.style.display = 'block';
  el.innerHTML = `
    <h1>LensForge needs WebGPU</h1>
    <p>LensForge integrates light rays through curved spacetime on your GPU and
       requires a browser with WebGPU support.</p>
    <p><code>${detail}</code></p>
    <p>Supported desktop browsers: Chrome/Edge 113+, Firefox 141+, Safari 26+.
       If your browser is recent, check <code>chrome://gpu</code> (or the
       equivalent) — the GPU may be blocklisted by the browser.</p>
  `;
}
