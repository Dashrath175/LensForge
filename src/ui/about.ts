/**
 * About & accuracy overlay (IMPLEMENTATION.md M8): the accuracy boundaries of
 * the simulation surfaced in-product, in plain language, with the data
 * attribution the shipped assets require. Content mirrors PHYSICS.md §10 and
 * VALIDATION.md §7 — the UI must not claim more than the validation record
 * supports (MASTER_PLAN.md principle 3: approximations are contracts).
 */
import { APP_VERSION } from '../core/version';

const STYLE_ID = 'lensforge-about-style';
const CSS = `
#lensforge-about {
  position: fixed; inset: 0; z-index: 40; display: flex;
  align-items: center; justify-content: center;
  background: rgba(0, 0, 0, 0.72);
}
#lensforge-about .about-card {
  max-width: 44rem; max-height: 84vh; overflow-y: auto;
  margin: 0 1rem; padding: 1.6rem 2rem; border-radius: 8px;
  background: #101418; border: 1px solid #2a3138;
  font: 13.5px/1.65 system-ui, sans-serif; color: #c8ced4;
}
#lensforge-about h1 { margin: 0 0 0.2rem; font-size: 1.25rem; font-weight: 600; color: #fff; }
#lensforge-about h2 { margin: 1.1rem 0 0.35rem; font-size: 0.95rem; font-weight: 600; color: #e8eef4; }
#lensforge-about .version { color: #6b747d; font-size: 0.85rem; }
#lensforge-about ul { margin: 0.25rem 0 0; padding-left: 1.2rem; }
#lensforge-about li { margin: 0.18rem 0; }
#lensforge-about .dim { color: #8a939c; }
#lensforge-about code { font: 12px ui-monospace, monospace; color: #e8b46a; }
#lensforge-about .close-button {
  float: right; margin: -0.4rem -0.8rem 0 1rem; padding: 0.3rem 0.7rem;
  background: none; border: 1px solid #3a424a; border-radius: 5px;
  color: #c8ced4; font: 13px system-ui, sans-serif; cursor: pointer;
}
#lensforge-about .close-button:hover { background: #1c232a; }
`;

const HTML = `
<div class="about-card" role="dialog" aria-modal="true" aria-label="About LensForge">
  <button class="close-button" type="button">Close</button>
  <h1>LensForge</h1>
  <div class="version">v${APP_VERSION} — physically rigorous Kerr black-hole renderer</div>

  <h2>What you are seeing</h2>
  <p>
    Every pixel is a null geodesic of the Kerr metric, integrated numerically
    (adaptive Dormand–Prince 4(5)) on the GPU, with exact relativistic
    radiometry: the accretion disk is the Novikov–Thorne (Page–Thorne) thin
    disk, shaded as a blackbody at the exactly redshifted/beamed temperature
    <code>T_obs = g·T_eff</code>; the sky is the real celestial sphere
    (lensed star positions, brightnesses, and magnifications are computed,
    not painted). Shadow, photon ring, Einstein rings, frame dragging,
    beaming asymmetry, and gravitational redshift all emerge from the
    equations — nothing is faked for artistic effect.
  </p>

  <h2>How it is validated</h2>
  <ul>
    <li>A float64 CPU oracle implements the same physics and is tested against
      closed forms (metric identities, critical impact parameter, weak-field
      deflection, ISCO/photon orbits, Page–Thorne flux) on every commit.</li>
    <li>The float32 GPU integrator is confronted with the oracle on identical
      rays: escape directions agree to &lt; 1 arcminute (High quality) away
      from the critical curve; disk-hit redshift factors to
      <code>Δg/g ≲ 10⁻⁴</code>.</li>
    <li>Rendered images are checked against the literature: the shadow
      silhouette lies within half a pixel of Bardeen's analytic contour at
      spins up to a = 0.998; disk-image geometry and beaming asymmetry match
      the float64 oracle at the Luminet (1979) and DNGR framings.</li>
  </ul>

  <h2>Accuracy boundaries</h2>
  <ul>
    <li><b>Photon subrings:</b> images of order n ≥ 2 are below the float32
      impact-parameter resolution and are not resolved.</li>
    <li><b>Near-critical rays:</b> within ~10⁻² of the critical impact
      parameter, sensitivity is physically chaotic; deviations there are
      measured and reported, not bounded.</li>
    <li><b>Disk model:</b> time-averaged, axisymmetric Novikov–Thorne thin
      disk — no GRMHD turbulence, no self-irradiation. The optional
      “turbulence” slider is a statistical texture advected at the exact
      orbital velocity Ω(r); it is not a plasma simulation.</li>
    <li><b>No polarization, no plasma effects</b> (synchrotron, absorption,
      scattering) — the disk emits as a blackbody.</li>
    <li><b>Bloom</b> (off by default) is a documented display/camera effect,
      not physics; tiled PNG exports render with bloom off.</li>
    <li><b>Orientation:</b> the black hole's orientation against the real sky
      is a fixed convention (spin axis toward the north celestial pole) —
      no astronomical black hole is depicted.</li>
    <li><b>Numerics:</b> images differ across GPUs at the least significant
      bit (driver FMA contraction); exports are 8-bit PNG through the
      float16 HDR pipeline.</li>
  </ul>

  <h2>Data &amp; licensing</h2>
  <ul>
    <li>Sky panorama: NASA/GSFC Scientific Visualization Studio,
      <i>Deep Star Maps 2020</i> (public domain).</li>
    <li>Bright stars: Yale Bright Star Catalogue, 5th edition — measured
      photometrically in the panorama and re-emitted as analytic point
      sources in one consistent radiometric system.</li>
  </ul>

  <p class="dim">
    Full derivations, tolerances, and measured results:
    <code>PHYSICS.md</code> and <code>VALIDATION.md</code> in the repository.
  </p>
</div>
`;

/** Show the overlay (idempotent — one instance at a time). */
export function showAbout(): void {
  if (document.getElementById('lensforge-about')) return;
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = CSS;
    document.head.appendChild(style);
  }
  const overlay = document.createElement('div');
  overlay.id = 'lensforge-about';
  overlay.innerHTML = HTML;
  const close = (): void => {
    overlay.remove();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (ev: KeyboardEvent): void => {
    if (ev.key === 'Escape') close();
  };
  overlay.addEventListener('click', (ev) => {
    if (ev.target === overlay) close();
  });
  overlay.querySelector('.close-button')?.addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
}
