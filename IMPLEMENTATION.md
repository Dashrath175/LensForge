# LensForge — Implementation Roadmap

Nine milestones, strictly ordered by dependency. Test IDs (`V-*`) are defined in VALIDATION.md; design decisions (`D#`) in ARCHITECTURE.md; equations (§) in PHYSICS.md. A milestone is **done** only when its exit criteria hold — demos are not exits.

Difficulty scale: ▲ low, ▲▲ moderate, ▲▲▲ high (dominated by correctness risk, not code volume).

---

## M0 — Foundations

- **Purpose:** A repository in which every later milestone can be built, tested, and deployed without infrastructure work.
- **Objectives:** Project scaffold (Vite + TypeScript strict), WebGPU device layer with capability tiering and device-lost recovery, WGSL preprocessor plugin, HDR render target → tone map → present path, CI pipeline, Vercel preview deploys.
- **Deliverables:** `src/app`, `src/gpu` skeleton, `src/shaders/post` minimal, `.github/workflows/ci.yml`, capability fallback page.
- **Dependencies:** none.
- **Validation criteria:** CI green (typecheck, empty test suite, build); manual: HDR gradient renders through AgX on Chrome/Edge/Firefox; unplugging adapter (device-lost simulation) recovers.
- **Exit criteria:** `npm run dev`, `npm test`, `npm run build` all work; PR preview deploy live; timestamp-query profiler shows pass timings.
- **Difficulty:** ▲
- **Implementation notes:** Build the WGSL `#include` preprocessor now — every later shader depends on it. Wire specialization constants (quality tier, step budget `K`) even though unused yet.
- **Risks:** Underestimating device-lost handling; retrofitting it later touches every pipeline. Do it first.

## M1 — CPU Physics Oracle (float64)

- **Purpose:** The scientific spine (MASTER_PLAN.md §1.6): a trusted float64 implementation of every equation in PHYSICS.md §2–§8.
- **Objectives:** Kerr inverse metric + analytic gradients; Hamiltonian RHS; DP45 with PI controller and monitors; tetrads (static, ZAMO, circular); circular-orbit kinematics; Page–Thorne flux integral; redshift factor; disk-intersection root-finding.
- **Deliverables:** `src/physics` complete with unit tests; `tools/` scripts generating the NT flux LUT and Planck→XYZ LUT.
- **Dependencies:** M0 (CI only).
- **Validation criteria:** V-A1 through V-A7 pass.
- **Exit criteria:** All V-A green in CI; a headless script renders a low-res (e.g. 128²) Schwarzschild shadow on CPU as an end-to-end smoke test; LUT generation reproducible.
- **Difficulty:** ▲▲▲
- **Implementation notes:** Write metric gradients analytically, not by finite difference — the WGSL port must be a literal translation. Keep every function pure and scalar-type-parameterizable where cheap (future df64). The CPU shadow render doubles as the reference image generator for V-O and V-I.
- **Risks:** Sign/convention errors that self-consistently cancel and pass weak tests — this is why V-A3/V-A4/V-A5 test *independent* closed forms.

## M2 — GPU Schwarzschild Lensing

- **Purpose:** First real image: lensed celestial sphere with shadow and Einstein ring, and the first GPU-vs-oracle confrontation.
- **Objectives:** Port metric/integrator/tetrad WGSL kernels (`a = 0` exercised, code is general Kerr); passes A–C with placeholder sky (procedural grid + low-res panorama); chunked march skeleton (D6) — correct, not yet tuned; `src/validation` harness comparing GPU rays to oracle.
- **Deliverables:** Live lensed sky at interactive rates; V-O harness with published deviation curves.
- **Dependencies:** M1.
- **Validation criteria:** V-O1, V-O4; V-I4 (Einstein ring); visual: shadow diameter matches `2b_c` at known FOV/distance.
- **Exit criteria:** V-O1 deviation curve within VALIDATION.md tolerances across the impact-parameter sweep including near-critical rays; no dispatch exceeds the V-P4 duration bound.
- **Difficulty:** ▲▲▲
- **Implementation notes:** The chunk scheduler and persistent SoA ray state are built here even though performance tuning is M6 — retrofitting checkpointability is the most expensive possible refactor (D6).
- **Risks:** f32 step-controller misbehavior near `b_c` (oscillating accept/reject). Mitigate with step clamps from PHYSICS.md §10 and the monitor-reject path; escalate to formulation review (D4 fallback) only with V-O evidence.

## M3 — Kerr

- **Purpose:** Full spin: frame dragging, asymmetric shadow, spin-dependent ISCO.
- **Objectives:** Nonzero `a` through the whole chain; ZAMO default observer; spin slider `a ∈ [0, 0.998]`; ergosphere-aware static-observer guard.
- **Deliverables:** Interactive spin control with correct shadow deformation.
- **Dependencies:** M2.
- **Validation criteria:** V-A5 (already green) plus V-I1 at `a ∈ {0, 0.5, 0.9, 0.998}`; V-O1/V-O4 re-run at high spin.
- **Exit criteria:** V-I1 Bardeen-contour overlay within pixel tolerance at all four spins; V-O curves at `a = 0.998` within tier tolerances.
- **Difficulty:** ▲▲
- **Implementation notes:** Little new code if M1/M2 kept the Kerr terms live at `a = 0` — this milestone is primarily a validation campaign. Cap at `a = 0.998` (Thorne limit; also keeps `Δ` cancellation manageable in f32).
- **Risks:** Polar-axis coordinate hazard (`sin θ → 0` in `g^φφ`): rays passing near the axis need the standard `θ`-reflection handling; test explicitly with polar camera positions.

## M4 — Thin Disk + Radiometry

- **Purpose:** The accretion disk with exact relativistic radiometry — the scientific heart of the project.
- **Objectives:** Disk intersection in Pass B; NT flux LUT + `T_obs = g·T_eff` blackbody shading (PHYSICS.md §8); Planck→XYZ→sRGB; HDR accumulation with exposure; `Ṁ` and `r_out` parameters; physical vs temperature-scaled color modes.
- **Deliverables:** Full disk render with beaming asymmetry, gravitational redshift, and lensed secondary/tertiary disk images.
- **Dependencies:** M3 (spin-dependent ISCO), M1 (LUTs).
- **Validation criteria:** V-A6, V-A7, V-O2, V-I2 (Luminet), V-I3 (DNGR).
- **Exit criteria:** V-I2 and V-I3 overlays within tolerance; beaming ratio across the disk matches oracle to V-O2 tolerance; energy bookkeeping audit (VALIDATION.md §2) signed off.
- **Difficulty:** ▲▲▲
- **Implementation notes:** A factor-of-`g` bug produces a plausible-looking wrong image — the literature overlays are the only defense; do them *during* development, not after. Beaming stays on (MASTER_PLAN.md §4).
- **Risks:** HDR range overwhelming rgba16f near high-spin ISCO (`g` large): accumulate in rgba32f if V-I3 shows quantization; radiometric calibration errors (silent wrongness — highest-severity project risk).

## M5 — Camera & Presets

- **Purpose:** Scientific usability: controlled observers, orbits, reproducible states.
- **Objectives:** Orbit + free camera; observer type selection (ZAMO/static/circular-orbit with validity guards); FOV/distance controls; time scaling via sim clock; preset JSON schema (full parameter + camera state); UI panel.
- **Deliverables:** `src/scene`, `src/sim`, `src/ui` complete; shipped preset set (e.g. "Luminet 1979", "DNGR Gargantua", "Face-on", "Photon-ring grazing").
- **Dependencies:** M4.
- **Validation criteria:** Presets byte-round-trip (save→load→save identical); circular-orbit camera at `r = 10` shows aberration consistent with oracle (spot-check via V-O1 rerun with boosted tetrad); accumulation restart/continue logic correct per ARCHITECTURE.md §6.
- **Exit criteria:** All controls live; time scaling affects camera orbit and disk phase parameter without accumulation bugs; presets reproduce V-I reference framings exactly.
- **Difficulty:** ▲
- **Implementation notes:** Presets are also test fixtures — V-I framings ship as presets so validation and product share one source of truth.
- **Risks:** Dirty-flag logic subtly wrong (stale accumulation after parameter change) — add an invariant check in dev builds: uniform hash vs accumulation tag.

## M6 — Performance

- **Purpose:** Hit MASTER_PLAN.md §10.4 without touching physics.
- **Objectives:** Tune chunk budget `K` from measured timings; dynamic internal resolution governor; temporal accumulation on camera rest (analytic jittered refinement); workgroup-size micro-benchmark at startup; profiler HUD; optional ray compaction only if justified (D8).
- **Deliverables:** Frame budget governor; profiler HUD; documented performance tiers.
- **Dependencies:** M5 (full feature surface to profile).
- **Validation criteria:** V-P1–V-P4; V-O1/V-O2 re-run at every quality tier (performance work must not silently change physics — tolerance per tier).
- **Exit criteria:** V-P1 (60 fps @1080p-equivalent, RTX 4070 Ti) sustained through a scripted camera path; V-P4 TDR bound never violated; still-frame convergence within V-P3 budget.
- **Difficulty:** ▲▲
- **Implementation notes:** Order of attack per ARCHITECTURE.md §5/§8: retirement → tolerance tiers → dynamic resolution → compaction (last, only with profiler evidence).
- **Risks:** Optimizations that bias the image (e.g. aggressive tolerance at screen edges) — every perf change re-runs V-O; nondeterministic timing flakiness in V-P (use medians over fixed scripted paths).

## M7 — Fidelity & Export

- **Purpose:** Final visual fidelity within the physics contract, plus export.
- **Objectives:** Production sky: Gaia/NASA panorama → KTX2 cubemap pipeline in `tools/` [R18][R19]; analytic bright-star splats with `|det J|⁻¹` magnification (D9); optional statistical disk inhomogeneity advected at `Ω(r)` (PHYSICS.md §9.6, off by default); optional documented bloom; PNG screenshot; tiled high-res export (to 8K) via the chunk scheduler.
- **Deliverables:** Shipping asset set within the 25 MB budget; export UI.
- **Dependencies:** M6 (export reuses the tuned scheduler).
- **Validation criteria:** V-I5 goldens re-baselined; star-splat magnification spot-checked against ray-bundle Jacobian from oracle; 8K export pixel-consistent with converged interactive frame (same physics, more samples).
- **Exit criteria:** No star twinkling/aliasing under lensing at any zoom; asset budget met; 8K export completes without TDR on target hardware.
- **Difficulty:** ▲▲
- **Implementation notes:** Attribute sky data in-app (licensing, MASTER_PLAN.md risk table analog). Inhomogeneity field must be clearly labeled "statistical model" in UI.
- **Risks:** Jacobian finite-differencing noisy at shadow edge — clamp magnification and document; KTX2 tooling friction on Windows (pin tool versions in `tools/`).

## M8 — Validation & Ship

- **Purpose:** Convert "it works" into "it is verified," then release.
- **Objectives:** Full V-* suite automated where possible; comparison figures generated into `docs/`; accuracy boundaries surfaced in UI (about panel); production deploy on tag.
- **Deliverables:** Complete VALIDATION.md results tables; release v1.0 on Vercel production.
- **Dependencies:** M7.
- **Validation criteria:** Every V-* test green or formally waived with rationale recorded in VALIDATION.md §Known Limitations.
- **Exit criteria:** MASTER_PLAN.md §10 success criteria all satisfied; a physicist-auditor walkthrough of PHYSICS.md + VALIDATION.md finds no unverifiable claim.
- **Difficulty:** ▲
- **Implementation notes:** Budget real time for writing up comparison figures — they are the project's scientific credibility.
- **Risks:** Discovery of a late radiometric discrepancy against [R5]; schedule assumes one re-validation loop back into M4-territory fixes.

---

**Total estimate:** 13–16 focused engineer-weeks. The three highest-risk items, in order: f32 integrator robustness near the photon sphere (M2), chunk-scheduler correctness (M2/M6), end-to-end radiometric calibration (M4).
