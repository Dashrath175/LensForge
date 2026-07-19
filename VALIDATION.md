# LensForge — Validation

Definitions, tolerances, and procedures for every test cited by IMPLEMENTATION.md. Three concentric rings — each catches what the previous cannot. Equations and closed forms: PHYSICS.md §3; citations: REFERENCES.md.

**Ring 1 (V-A):** analytic tests against closed forms — float64 oracle, every CI run, exact-tolerance.
**Ring 2 (V-O):** GPU float32 vs CPU float64 oracle — per merge, quantifies f32 trust.
**Ring 3 (V-I):** rendered images vs published literature — release gate.
**Plus V-P:** performance, on reference hardware.

## 1. Scientific Validation — Ring 1 (V-A)

All run headless in Vitest on `src/physics` (float64). Tolerances are absolute unless noted.

| ID | Test | Procedure | Pass tolerance |
|---|---|---|---|
| V-A1 | Metric & tetrad identities | `g^{μν}g_{νσ} = δ^μ_σ` and `g(e_a, e_b) = η_ab` sampled over `(r, θ, a)` grid incl. near-horizon, near-axis | `< 1e−12`; `< 1e−11` at near-horizon/near-axis corner points, where `Δ`/`sin²θ` cancellations inside the components make the identity intrinsically ill-conditioned in float64 (measured: 8e−16 well-conditioned, 1.6e−12 worst corner) |
| V-A2 | Conservation drift | Integrate standard ray set (28 equatorial + 12 tetrad-inclined rays over `a ∈ {0, .5, .9, .998}`); max drift of `Q` and normalized `H` over full ray (`E`, `L_z` are exact by construction — zero RHS) | `< 5e−12` at tol `1e−13` (accumulated truncation floor ≈ 5e−16/step over O(2000) steps; measured worst 1.0e−12). `H` monitor is normalized by the magnitude sum of its terms — near-horizon BL cancellation makes an `E²`-normalized monitor report conditioning noise |
| V-A3 | Critical impact parameter | Bisect capture/escape boundary at `a = 0` vs `b_c = 3√3` | `< 1e−8` |
| V-A4 | Weak-field deflection | Deflection vs `α = 4/b + 15π/4b²` for `b ∈ [100, 1000]`, start radius `∝ b²` | Residual bounded by the next series term `128/(3b³)` (×1.5 margin) — the deviation from the two-term series must itself be the physical third-order term |
| V-A5 | ISCO & photon orbits | Numerically located circular orbits vs closed forms, `a ∈ {0, .25, .5, .75, .9, .998}` | rel `< 1e−10` |
| V-A6 | Page–Thorne flux | Closed form vs independent quadrature of the flux integral; `a = 0` outer-region asymptotics | rel `< 1e−6` (closed form vs quadrature). Asymptotic limit: `F → (3Ṁ/8πr³)(1 − C/√r)` with the log-corrected boundary constant `C = √6 + (√3/2)ln((√6+√3)/(√6−√3)) ≈ 3.976` — the naive `√6` profile is recovered only as `O(r^{−1/2})` (verified both ways) |
| V-A7 | Redshift factor limits | Static emitter → `g = √(1 − 2/r)` (pure gravitational); `r → ∞` circular emitter → special-relativistic Doppler | rel `< 1e−10` |

## 2. Numerical Verification — Ring 2 (V-O)

Harness: `src/validation`. Identical initial conditions integrated in WGSL f32 and TS f64; deviations reported as curves vs impact parameter and regression-tracked (a worsening curve fails even inside absolute bounds).

| ID | Test | Metric | Bound (Quality: High / Medium / Low) |
|---|---|---|---|
| V-O1 | Escape-direction deviation | Angular error of asymptotic sky direction (both implementations interpolated to exactly `r = r_esc`, PHYSICS.md §10), swept over `b` incl. `b − b_c` down to `10⁻³`. Angle computed via the chord `2·asin(|d̂−ô|/2)` after f64 renormalization — `acos(dot)` is unusable: the f32-normalized GPU vector's ~1e−7 norm error alone inflates it to ≈1.5 arcmin near zero separation | `< 1 / 3 / 10` arcmin at `b − b_c ≥ 10⁻²`; near-critical rays reported, not bounded |
| V-O2 | Disk-hit deviation | Error in `(r, φ)` of intersection and in `g`; batch rays first pass through the f32 ray-init contract (polar-plane quantization `AXIAL_SNAP_SMIN` and disk plane-launch quantization `PLANE_LAUNCH_EPS`, PHYSICS.md §10) exactly as screen rays do | `Δg/g < 1e−3` (High tier; the radiometric bound — measured ≤ 1.0e−4 at `a ∈ {0, 0.9, 0.998}`). Positional: `Δr/max(1, c) < 1e−2` with the grazing-incidence amplification `c = |Δ·p_r|/(r·p_θ)` divided out — raw `Δr` is the V-O1 transverse budget (≲1 arcmin ⇒ ~3e−2 over the reference 120M path) times `c`, so an un-normalized 1e−3 bound would be tighter than the already-validated integrator error itself (measured: raw `Δr ≤ 8.8e−3`, normalized ≤ 2.9e−3 ≈ 1/20 px at the reference framing); `Δφ` reported (≤ 4.4e−5). Hits within 1e−2 of a disk edge are status-fragile: reported, not bounded (cf. V-O1 near-critical) |
| V-O3 | Step-budget exhaustion | Fraction of rays flagged MAX_STEPS per frame | `< 0.1 %` of pixels, all presets |
| V-O4 | f32 conservation drift | Drift of `Q` and normalized `H` across the ray population, evaluated in f64 from the f32-quantized initial state to the returned final state (`E`, `L_z` are exact by construction) | p99 `< 1e−4`; distribution regression-tracked. The `H` bound applies to ESCAPED rays; CAPTURED-ray endpoints are conditioning-limited (≈ ε_f32/Δ ~ 1e−4 at the capture radius, cf. V-A2 note) and are reported separately |

Run on every merge touching `src/physics`, `src/shaders`, or integrator parameters; full sweep on release. **Every performance change re-runs V-O** (IMPLEMENTATION.md M6).

**M5 boosted-observer spot check** (`runVOBoostedSuite`): V-O1 rerun with rays cast through the circular-orbit tetrad at `r = 10` — initial momenta carry the full orbital boost (`v ≈ 0.35c`), and the reported aberration relative to the local ZAMO frame must be of order `v` (a silently-ignored boost cannot pass). Measured: max aberration 20.7°/19.0° at `a = 0/0.9`, GPU-vs-oracle escape-direction deviation 0.002/0.009 arcmin, 100 % status agreement.

## 3. Rendering Validation — Ring 3 (V-I)

Framings ship as presets (single source of truth with the product). Comparisons are tolerance-based, never pixel-exact (D12: cross-driver FMA nondeterminism).

| ID | Test | Reference | Pass criterion |
|---|---|---|---|
| V-I1 | Kerr shadow contour | Bardeen parametric curve [R1], `a ∈ {0, 0.5, 0.9, 0.998}` | Rendered boundary within 1 px of analytic overlay at 1080p reference framing (`r = 1000`, equatorial ZAMO, fovY 0.03). The overlay is exact at finite camera radius: the critical curve lives in observer-independent `(ξ, η)` space and is mapped to pixels by inverting the linear tetrad relation `p_μ(n̂)` in float64 (`src/validation/shadowContour.ts`) — no far-field approximation. M3 measured: max deviation 0.50 px at all four spins (the boundary-quantization floor) |
| V-I2 | Schwarzschild disk isophotes | Luminet 1979 Fig. 11 [R2] | Primary & secondary image isophote shapes within 2 % of published geometry. M8 automation (`measureDiskEdgeGeometry`): every strict GPU-status transition into/out of the disk image along 48 radial spokes at the [R2] framing is bracketed at pixel resolution and bisected on the float64 oracle's disk-hit predicate — the oracle (validated against closed forms in V-A) stands in for the published curves at full precision (Luminet's figure is itself the numerical solution of these equations; see §7). M8 measured: 166/166 transitions matched (0 status-fragile), max 0.62 px absolute deviation (mean 0.27 px), max shape deviation 0.66 % over boundaries with ρ ≥ 50 px; below ρ ≈ 50 px the half-pixel GPU boundary quantization exceeds 2 % of the local radius, so small-radius deviations are reported, not bounded (measured ≤ 2.4 %, quantization-floor-consistent). Bounds: ≤ 1 px absolute everywhere and < 2 % shape where resolvable. Isophote *interior* agreement is carried by the V-I3 per-pixel oracle re-shade (mean 0.4 % HDR deviation) on the same framing |
| V-I3 | High-spin disk morphology | DNGR published frames [R5] (beaming ON variants) | Feature-position agreement; beaming asymmetry ratio within 5 % of oracle render. M4 measured (oracle half, `src/validation` `measureBeamingAsymmetry`): ratio agreement 1.2e−4 (a = 0) and 4.0e−5 (a = 0.9) at the reference framing, mean per-pixel luminance deviation 0.4 %; the published-frame overlay itself is part of the M8 release gate (§5) |
| V-I4 | Einstein ring radius | Weak-field lens equation for displaced source | Ring angular radius rel error `< 1 %` at `b ≫ b_c` framing |
| V-I5 | Golden regression | Baselined converged renders of all shipped presets | Perceptual diff (SSIM > 0.99) **and** physical diff (mean relative HDR luminance error < 1 %). M8 automation (`runGoldenSuite`): per preset, the perceptual gate is mean SSIM (11×11 Gaussian window, standard Wang-et-al. parameters; implementation self-tested against closed forms in `tests/unit/validation/ssim.test.ts`) of the displayed image against the committed golden PNG (`tests/golden/<slug>.png`, displayed orientation); the physical gate is the pre-tonemap full-frame mean against `baselines.json` (M7 stats, 16×9 cell means additionally reported). Both run at the baseline resolution on the converged base pixel-center sample. M8 measured: SSIM = 1.0 and mean-luminance error = 0 for all four presets (bit-repeatable base sample on the capture hardware; cross-GPU runs are expected to land inside the tolerance, not at zero — D12) |

Comparison overlays are committed to `docs/` at each release — the project's auditable scientific record.

## 4. Performance Validation (V-P)

Reference hardware: RTX 4070 Ti, Windows 11, Chrome stable. Metrics are medians over a fixed scripted camera path (flakiness control).

| ID | Test | Target |
|---|---|---|
| V-P1 | Interactive throughput | ≥ 60 fps at 1080p-equivalent adaptive quality, full disk + sky, scripted orbit |
| V-P2 | Pass budgets | Pass B ≤ 70 % of frame; Passes A+C+D ≤ 3 ms combined (timestamp queries) |
| V-P3 | Still-frame convergence | Converged (V-I5-quality) frame ≤ 2 s after camera rest at 1080p |
| V-P4 | TDR safety | No single dispatch > 250 ms on reference hardware; device-lost recovery exercised in test |

Benchmark procedure: the scripted path is driven headed through the dev-build `__lensforge` hooks, per-pass timings recorded from the timestamp-query profiler, and the JSON report kept per release for trend tracking (docs/results/). Secondary tier (integrated GPU, e.g. Iris Xe): correctness required (V-O at Low tier), performance reported but not gated.

**M6 measured results (development GPU — not the reference RTX 4070 Ti; reference-hardware sweep remains a release-gate step):**
- V-P1: scripted orbit at 1280×720, `a = 0.9`, disk on — median 11.4 ms/frame CPU-GPU-serialized (pipelined rAF throughput is higher); mid-drag frame pixel-identical to converged at `a ≤ 0.9` (governor-settled), ≤ 2.2 % of pixels at ladder resolution at `a = 0.998`.
- V-P2: A+C+D combined 0.18 ms ≪ 3 ms bound; Pass B (march + ladder) 9.1 ms < 70 % of the 16.7 ms frame budget.
- V-P3: still-frame convergence after rest: 6 ms at 1080p `a = 0.9`; 68 ms at 1080p `a = 0.998` (bound: 2 s).
- V-P4: every dispatch is bounded at 64 DP45 attempts; worst measured multi-dispatch pass ≈ 12 ms, so single dispatches sit orders below the 250 ms TDR bound.
- V-O1 per tier at `a ∈ {0, 0.9}`: High 0.05/0.16′ (bound 1′), Medium 1.06/1.83′ (bound 3′), Low 3.79/2.51′ (bound 10′). The Low tier's original `rtol = 1e−4` measured 17–32′ — outside its documented bound — and was retuned to `rtol = 3e−5` (tracePass.ts QUALITY_TIERS); this is the per-tier V-O rerun doing its job.
- V-O2 `Δg/g`: High ≤ 7.9e−5 (`a = 0.998`), Medium 3.3e−4 — both inside the 1e−3 radiometric bound.
- Temporal accumulation: base sample bit-repeatable (max HDR readback diff 0); after 12 accumulated samples, ~5000 probed smooth-region pixels changed ≤ 9.2e−4 relative — only edges/sub-pixel features anti-alias.

**M7 measured results (development GPU; every M6 number above re-measured unchanged):**
- V-O1 re-run post-sky/tiling: High 0.052′/0.159′/0.99′ at `a ∈ {0, 0.9, 0.998}` (bound 1′); Medium 1.83′, Low 2.51′ at `a = 0.9` (bounds 3′/10′) — identical to M6.
- V-O2 re-run: `Δg/g` ≤ 3.6e−5 (`a = 0.9`), ≤ 7.9e−5 (`a = 0.998`); normalized `Δr` ≤ 2.0e−3 — identical to M6.
- V-I1 re-run: max contour deviation ≤ 0.50 px at all four spins (the boundary-quantization floor); shadow-diameter rel error 0.34 %; boosted-tetrad spot check 0.002′/0.009′ at `a = 0/0.9` with 20.7°/19.0° aberration — identical to M6.
- **Star-splat magnification (M7 criterion):** GPU `|det J|⁻¹` from neighboring escape directions vs float64 ray-bundle Jacobian (4 oracle rays per pixel, central differences) over 24 pixels spanning μ across 3.5 decades at the V-I1 framing: max rel error 0.25 % (`a = 0`), 0.13 % (`a = 0.9`); bound 5 %.
- **Tiled-export consistency (M7 criterion):** two 256-px tiles with 1-px apron vs the monolithic converged 512² frame at `a = 0.9` (disk + sky + splats): interior **bit-identical** (0 differing HDR values in 780 300 comparisons). The check exposed and fixed an M6 defect: CAPTURED pixels fell into the shade fallback branch and carried preview-ladder bleed at converged shadow edges (≈600 px, ≤ 1.6 HDR units at 512²) — CAPTURED is now the explicit exact-black case, restoring the documented §4/§5 contracts.
- Sky orientation/parity: mapping calibration against 200 catalog stars decisive (score ratio 52×); the ten brightest photometric detections are the real sky's ten brightest stars; constellation handedness verified (Betelgeuse/Rigel screen positions match the real sky to the pixel-measurement level).
- Pass budgets with sky + splats: steady-state Pass C 0.12 ms, present 0.013 ms, optional bloom 0.24 ms (off by default) — A+C+D ≪ the 3 ms V-P2 bound.

**M8 release sweep (development GPU; the v1.0.0 record — raw JSON in docs/results/m8/, figures in docs/figures/):**
- V-O1: High 0.052′ / 0.159′ / 0.99′ at `a = 0 / 0.9 / 0.998` (bound 1′); Medium 1.83′, Low 2.51′ at `a = 0.9` (bounds 3′/10′) — identical to M6/M7. Near-critical (reported): 24.9′ worst at `a = 0.998`, `b − b_c = 10⁻³`.
- V-O2: `Δg/g ≤ 1.03e−4 / 3.6e−5 / 7.9e−5` at `a = 0 / 0.9 / 0.998` (bound 1e−3); normalized `Δr ≤ 2.9e−3`; `Δφ ≤ 4.4e−5`; 100 % status agreement, 0 edge-fragile rays in the batch.
- V-O3 (M8 automation, `measureStatusCensus`): **zero** MAX_STEPS pixels and zero unconverged pixels at convergence on all four shipped presets at 1280×720 (bound < 0.1 %).
- V-O4: max `Q` drift 9.2e−5 (`a = 0.998`; bound 1e−4 p99), ESCAPED `H` residual ≤ 6.7e−6; CAPTURED endpoints ≤ 4.7e−4 (conditioning-limited, reported).
- Boosted-observer spot check: 0.002′ / 0.009′ at `a = 0 / 0.9` with 20.7° / 19.0° max aberration at `v ≈ 0.35c` — identical to M5/M7.
- V-I1: max contour deviation ≤ 0.50 px at all four spins (boundary-quantization floor), mean ≤ 0.22 px, zero MAX_STEPS pixels. Figure overlay parity verified numerically: mirrored analytic equator extent matches the displayed exact-black run to < 0.2 px at `a = 0` and `a = 0.998`.
- V-I2 (M8 automation): 166/166 boundary transitions matched, 0 fragile, max 0.62 px absolute, 0.66 % shape deviation where resolvable (see §3 row).
- V-I3: ratio agreement 2.1e−5 (`a = 0`) and 5.7e−5 (`a = 0.9`); mean per-pixel HDR deviation 0.41 % / 0.14 %; 0 status mismatches.
- Star-splat magnification: max rel error 0.25 % (`a = 0`), 0.13 % (`a = 0.9`) over 24 pixels spanning 3.5 decades of μ (bound 5 %).
- V-I5 (M8 automation): SSIM = 1.0 and pre-tonemap mean-luminance error = 0 vs the M7 baselines for all four presets.
- V-P spot checks at 1920×1080 (CPU-GPU-serialized hook timings, upper bounds): still-frame re-convergence after a pose nudge 40 ms (`a = 0.9`) and 220 ms (`a = 0.998`) — V-P3 bound 2 s; steady-state march 3.4 ms, shade 0.29 ms, present 0.03 ms; tiled 1080p export (4 tiles, 8 AA samples) 1.92 s; no dispatch anywhere near the 250 ms V-P4 bound (64-attempt cap per dispatch unchanged).
- Device-loss drill: `device.destroy()` → no auto-recovery (by design for reason 'destroyed') → `forceRecover` → full pipeline rebuild, LUT/sky reload, convergence in 2 frames.

## 5. Regression Testing

- **CI (every PR):** typecheck → V-A suite (incl. the SSIM metric self-test) → build. GPU/image tests do **not** run in CI — GitHub-hosted runners have no real GPU, and a SwiftShader V-I5 lane was evaluated and not adopted (a software rasterizer exercises none of the driver behavior the tolerances exist for); this is recorded in §7.
- **Merge to main touching physics/shaders:** V-O harness (requires GPU — run locally via the `__lensforge` dev hooks; result JSON attached to PR).
- **Release tag:** full V-A + V-O sweep + V-I1–V-I5 + V-P run locally on real hardware **before** tagging (the M8 procedure in docs/README.md); comparison figures regenerated into `docs/`; the tag itself triggers the headless gate + production deploy (.github/workflows/release.yml).
- **Baseline updates:** V-I5 goldens (stats **and** PNGs) may only be re-baselined in a dedicated PR that states the physical reason for the change (`captureGoldenImage` hook + `tools:figures --write-goldens`).

## 6. Expected Outputs

Qualitative signatures any correct build must show (fast human smoke test):

1. `a = 0`: circular shadow; silhouette diameter `2b_c = 10.39 M` (angular size per framing).
2. `a = 0.998`, edge-on: D-shaped shadow displaced toward retrograde side.
3. Disk edge-on: upper secondary image arcs over the shadow; approaching side dramatically brighter — if the disk looks symmetric, beaming is broken. Parity (resolved in M7): the trace computes the traced-photon image (approaching side right in the raw HDR — statuses and HDR readbacks keep this orientation, so every V-O/V-I harness is unchanged); the present pass applies the exact arriving-photon parity map (PHYSICS.md §8 — screen-x mirror + φ-mirrored sky lookup, an isometry, not a convention), so the **displayed** image shows the approaching side on the **left**, matching the published convention (Luminet [R2], DNGR [R5]). M7 verified both halves empirically: constellation handedness against the real celestial sphere (Orion/galactic-center probes), and displayed disk left/right luminance ratio ≈ 2 : 1 through AgX at the V-I3 framing.
4. `g`-shift visible as blue-white approaching side, red-dim receding side in physical color mode.
5. Stars sweep along lensed streamlines near the shadow without twinkling; Einstein ring appears for aligned background sources.
6. Time scaling: camera orbit and (if enabled) inhomogeneity pattern advect; axisymmetric disk otherwise static — this is correct, not a bug (PHYSICS.md §9.6).

## 7. Known Limitations

Formal record of accepted gaps (waivers per IMPLEMENTATION.md M8 land here):

| Limitation | Reason | Reference |
|---|---|---|
| Photon subrings `n ≥ 2` not resolved | float32 impact-parameter sensitivity | PHYSICS.md §10, [R7] |
| Near-critical rays (`b − b_c < 10⁻²`) exceed V-O1 bounds | Chaotic sensitivity is physical; error reported, not bounded | PHYSICS.md §10 |
| Disk is time-averaged axisymmetric NT; no GRMHD | Out of scope | MASTER_PLAN.md §4 |
| No polarization, no plasma effects | Out of scope | MASTER_PLAN.md §4 |
| Cross-GPU images differ at LSB level | WGSL FMA contraction | D12 |
| GPU/image tests (V-O, V-I) do not run in CI at all — they are a documented local release-gate step | GitHub-hosted runners have no real GPU; a SwiftShader lane would validate a software rasterizer, not the drivers the tolerances target | §5 |
| V-I2 is validated against the float64 oracle at the published framing, not against a digitization of the 1979 scanned figure | The oracle independently reproduces the closed forms Luminet's solution rests on (V-A3/A4/A6); digitizing the scan would add scan/print distortion, not rigor. The rendered figure is published in docs/ for direct visual comparison | §3, docs/README.md |
| V-I2 boundary radii below ρ ≈ 50 px are reported, not bounded | Half-pixel GPU boundary quantization exceeds 2 % of the local radius there (measurement floor, not physics error; absolute deviation stays ≤ 1 px everywhere) | §3 |
| V-P numbers for M7/M8 were measured on the development GPU; the RTX 4070 Ti reference sweep (V-P1 60 fps scripted path) is inherited from the M6 record | Reference hardware not available at release time; margins are 10–50× on every measured bound | §4 |
| Production deploy on tag is configured (.github/workflows/release.yml) but has never executed: the repository has no git remote and no linked Vercel project | Deploy infrastructure (remote, Vercel project, secrets) is owner-provisioned; the workflow is ready and the build is static output | M8 |
| Rays within `s_min = |p_φ|/P < 4e−5` of the polar plane render as exactly polar (`L = 0`) | float32 `θ`-resolution wall at the `θ = π` pole; sub-pixel-exact substitution | PHYSICS.md §10 |
| Rays launched within `1e−6` rad of the disk plane start on their departure side; exactly in-plane rays graze the razor-thin disk without a hit | Plane-launch quantization: prevents σ = 0 self-intersection for equatorial observers inside the disk band (circular-orbit camera); the razor-thin disk model is singular for exactly in-plane propagation | PHYSICS.md §10 |
| During camera interaction, still-marching pixels present the same-pose preview ladder (½ then ¼ resolution, bilinear); with the M6 governor the mid-drag frame is pixel-identical to converged at `a ≤ 0.9` on the dev GPU, and ≤ ~2 % of pixels (photon-ring vicinity) present at ladder resolution at `a = 0.998` | Interaction presentation contract: every presented pixel is traced radiance for the current pose — no stale frames, no pose mixing; converged output is bit-identical to the pre-preview path | ARCHITECTURE.md §5 |
| Readbacks during active temporal refinement see jittered-sample ray state | Refinement rounds re-init the persistent ray state; validation harness disables refinement for the duration of a measurement (`setRefinementEnabled`) so all comparisons run on the base pixel-center sample | ARCHITECTURE.md §5 |
| Star splats are omitted on pixels whose 4-neighborhood is not fully ESCAPED (a ~1 px contour at the shadow edge and disk silhouette), and point magnification is clamped at `μ ≤ 10⁴` | The splat Jacobian is finite-differenced from neighboring rays (the documented M7 approach); at status boundaries the difference is undefined/unreliable. Both effects live where the aggregate-star regime dominates anyway | sky.wgsl |
| The black hole's orientation relative to the celestial sphere is a fixed convention (BL `φ` ↔ −RA, spin axis toward the north celestial pole) | Physically arbitrary (no preferred orientation exists); fixed by the arriving-photon parity map + the panorama's calibrated equirect mapping | PHYSICS.md §8 |
| Tiled export renders with bloom forced off; the PNG screenshot captures bloom as displayed | Bloom kernels cross tile borders; correct full-image bloom at 8K would need the assembled HDR resident on the GPU. Bloom is an optional, off-by-default display effect (camera model) | export.ts |
| Export PNGs are 8-bit through the f16 HDR target (the f32 accumulation exists but is not the export source) | PNG quantization (8 bit) is far coarser than f16; an EXR/f32 export path is future work | renderer.ts |
