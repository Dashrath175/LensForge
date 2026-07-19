# Changelog

All notable changes to LensForge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-07-19

First public release. Everything below was built across nine validated
milestones (M0–M8, see [IMPLEMENTATION.md](IMPLEMENTATION.md)); the complete
scientific release record lives in [docs/](docs/README.md).

### Physics & rendering

- Kerr null-geodesic integration on the GPU (WGSL compute): Hamiltonian
  formulation, adaptive Dormand–Prince 4(5) with PI step control, spin
  `a ∈ [0, 0.998]`, Boyer–Lindquist coordinates.
- Float64 CPU oracle (`src/physics/`) implementing identical physics,
  validated against closed forms (metric identities, critical impact
  parameter, weak-field deflection, ISCO/photon orbits, Page–Thorne flux) —
  70 unit tests, every commit.
- Novikov–Thorne thin accretion disk with exact relativistic radiometry:
  redshift factor `g` via tetrads, Doppler beaming, `I_ν/ν³` invariant,
  blackbody Planck → CIE XYZ → sRGB pipeline; physical and
  temperature-scaled color modes.
- Real celestial sphere: NASA Deep Star Maps 2020 cubemap with the Yale
  Bright Star Catalogue re-emitted as analytic point sources under
  `|det J|⁻¹` lensing magnification — star positions, brightnesses, and
  magnifications are computed, not painted.
- Observers: ZAMO, static (ergosphere-guarded), circular orbit (ISCO-clamped,
  full orbital boost aberration).
- HDR pipeline with AgX tone mapping; optional energy-conserving bloom
  (display-only, off by default); optional statistical disk inhomogeneity
  advected at the exact `Ω(r)` (labeled statistical model, not GRMHD).
- Progressive refinement to a converged still frame; chunked, TDR-safe
  marching; frame-budget governor; measured startup workgroup benchmark.
- PNG screenshot and tiled export to 8K — tiles are bit-identical to a
  monolithic render.

### Validation (v1.0.0 record — docs/results/m8/)

- GPU vs float64 oracle escape directions: 0.052′ / 0.159′ / 0.99′ at
  `a = 0 / 0.9 / 0.998` (High tier, bound 1′).
- Disk-hit redshift factor: `Δg/g ≤ 1.03e−4` (bound 1e−3).
- Rendered shadow vs Bardeen analytic contour: ≤ 0.50 px at four spins.
- Disk-image boundary geometry at the Luminet 1979 framing: 166/166
  transitions matched, ≤ 0.62 px.
- Beaming asymmetry vs full float64 re-shade: ≤ 5.7e−5.
- Golden-image regression: SSIM = 1.0, HDR mean error = 0.

### Application

- Tweakpane control panel (spacetime, camera, disk, display, export,
  presets), premium glass UI, loading sequence, `H` panel toggle.
- Four shipped presets doubling as validation framings: *Luminet 1979*,
  *DNGR Gargantua*, *Face-on*, *Photon-ring grazing*; canonical JSON preset
  save/load.
- In-app **About & accuracy** panel stating the measured accuracy boundaries.
- WebGPU capability page; automatic device-loss recovery.

[1.0.0]: https://github.com/OWNER/lensforge/releases/tag/v1.0.0
