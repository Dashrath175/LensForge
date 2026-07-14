# LensForge — Master Plan

Source of truth for scope, goals, and principles. If a proposed change conflicts with this document, the change is wrong or this document must be amended first (see DEVELOPMENT.md §Contribution Rules).

## 1. Philosophy

1. **Physics produces the image.** The renderer visualizes what the geodesic integrator and radiative transfer compute. No phenomenon is implemented as a standalone "effect."
2. **Correctness before performance, performance before features.** A fast wrong image is worthless; a correct slow image is a milestone.
3. **Approximations are contracts.** Every deviation from full general-relativistic radiative transfer is listed in PHYSICS.md §10 and surfaced honestly in the UI where user-visible.
4. **Simplify formulations, never physics.** Prefer the numerically simplest formulation of the exact physics (e.g. Hamiltonian geodesics over Christoffel expansion — decision D3).
5. **No AI in the render path.** No learned upscaling, denoising, or generation. Temporal accumulation and filtering must be analytic.
6. **The float64 CPU oracle is the scientific spine.** GPU kernels are validated float32 ports of `src/physics/`; disagreement beyond tolerance (V-O tests) blocks merge.

## 2. Scope

A real-time Kerr black hole renderer for modern desktop browsers: backward ray tracing of null geodesics from a camera through curved spacetime, shading escaped rays with a celestial sphere and disk-intersecting rays with a Novikov–Thorne thin accretion disk, through an HDR radiometric pipeline to the screen.

## 3. Goals

- Physically exact (within float32, see PHYSICS.md §10) strong-field lensing for arbitrary spin `a ∈ [0, 1)`.
- Exact relativistic radiometry: redshift, Doppler beaming, aberration via tetrads and the `I_ν/ν³` invariant.
- Interactive: 60 fps at 1080p-equivalent quality on an RTX 4070 Ti-class GPU; degraded-but-correct on integrated GPUs.
- Converged still-frame quality: progressive refinement when the camera rests; tiled export up to 8K.
- Fully static deployment (GitHub → Vercel), no backend.
- Scientific verifiability: every claim backed by a V-* test against published results.

## 4. Non-Goals

- GRMHD simulation or turbulent plasma dynamics.
- Polarized radiative transfer.
- Photon subrings of order n ≥ 2 in real time (float32 boundary, PHYSICS.md §10).
- Mobile-first support, WebGL2 fallback (D1).
- Cinematic imitation of *Interstellar* — DNGR [R5] is a validation reference, not an aesthetic target. Notably, beaming stays **on**.
- Game-engine features: meshes, scene graphs, physics beyond geodesics.

## 5. Feature List

| Feature | Mechanism (all emergent from two components: integrator + transfer) |
|---|---|
| Event horizon / shadow | Ray capture at `r → r₊` |
| Schwarzschild ↔ Kerr | Single Kerr engine; Schwarzschild is `a = 0` |
| Photon sphere / photon ring (n ≤ 1) | Geodesic integration |
| Gravitational lensing, Einstein ring, starfield distortion | Escaped rays sample celestial sphere |
| Gravitational redshift, Doppler beaming, aberration | Redshift factor `g` + tetrad ray init |
| Accretion disk | Novikov–Thorne thin disk, ISCO-truncated |
| Frame dragging | Kerr metric, automatic |
| Adjustable mass, spin, `Ṁ`, FOV, distance, time scale | Parameter store → uniforms |
| Camera orbit, observer selection | Tetrad construction per observer type |
| Physical vs temperature-scaled disk color | Planck → CIE XYZ pipeline, two calibrations (both labeled) |
| HDR + AgX tone mapping, exposure | Mandatory: beaming spans ~10³–10⁴ dynamic range |
| Optional statistical disk inhomogeneity | Advected at exact `Ω(r)`; off by default; labeled statistical model |
| Presets, screenshot export, tiled high-res export | Renderer chunk scheduler reuse |

## 6. Milestone Overview

Details, dependencies, exit criteria: IMPLEMENTATION.md.

| # | Milestone | One-line outcome |
|---|---|---|
| M0 | Foundations | Repo, CI, WebGPU device layer, HDR present path |
| M1 | CPU physics oracle | Validated float64 Kerr geodesics + disk physics |
| M2 | GPU Schwarzschild lensing | Live lensed sky, shadow, Einstein ring |
| M3 | Kerr | Spin, frame dragging, asymmetric shadow |
| M4 | Thin disk + radiometry | Novikov–Thorne disk with exact beaming/redshift, HDR |
| M5 | Camera & presets | Orbits, observers, FOV, time scaling, preset JSON |
| M6 | Performance | Chunked march, dynamic resolution, 60 fps target |
| M7 | Fidelity & export | Star-catalog splats, color modes, tiled 8K export |
| M8 | Validation & ship | Full V-* suite green, docs complete, production deploy |

## 7. Engineering Principles

- One fact, one owner: code has one module per responsibility (ARCHITECTURE.md), docs have one file per concern (README.md §Document Map).
- Physics changes land in `src/physics/` with unit tests **before** the WGSL port.
- All GPU work is chunk-schedulable — no dispatch may assume it completes a frame (D6, TDR constraint).
- Dependencies are liabilities: runtime deps limited to Tweakpane; adding one requires justification in a PR.
- Every quality/performance trade-off is a single explicit knob (integrator tolerance), not scattered constants.

## 8. Scientific Principles

- Geometrized units `G = c = M = 1` in all kernels (PHYSICS.md §1).
- Conserved quantities (`E`, `L_z`, `Q`, `H = 0`) are error monitors, never integration variables.
- Radiometric quantities carry physical meaning until tone mapping; the "camera model" (exposure, optional bloom) is the only artistic layer and is documented as such.
- Published-literature comparison (V-I tests) is a release gate, not a nice-to-have.

## 9. Constraints

- Browser only, static hosting, no backend, no server-side compute.
- WebGPU/WGSL: float32 only in kernels (df64 is future work).
- Windows GPU watchdog (~2 s): bounded work per dispatch (D6).
- Asset budget < 25 MB total download (dominated by sky map).
- Desktop browsers: Chrome/Edge ≥ 113, Firefox ≥ 141, Safari ≥ 26; no user-agent sniffing, feature detection only.

## 10. Success Criteria

1. All analytic tests V-A pass in CI on every commit.
2. GPU-vs-oracle deviation curves (V-O) within published tolerances for all quality tiers.
3. Rendered Kerr shadow matches Bardeen contour [R1], disk isophotes match Luminet [R2] and DNGR [R5] within V-I tolerances.
4. 60 fps at 1080p-equivalent adaptive quality on RTX 4070 Ti (V-P1).
5. 8K export produces a converged, physically identical (higher-sample) image via the same code path.
6. A physicist can audit every rendered phenomenon from PHYSICS.md + VALIDATION.md without reading code.
