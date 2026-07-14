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
| V-O2 | Disk-hit deviation | Error in `(r, φ)` of intersection and in `g` | `Δr < 1e−3`, `Δg/g < 1e−3` (High tier) |
| V-O3 | Step-budget exhaustion | Fraction of rays flagged MAX_STEPS per frame | `< 0.1 %` of pixels, all presets |
| V-O4 | f32 conservation drift | Drift of `Q` and normalized `H` across the ray population, evaluated in f64 from the f32-quantized initial state to the returned final state (`E`, `L_z` are exact by construction) | p99 `< 1e−4`; distribution regression-tracked. The `H` bound applies to ESCAPED rays; CAPTURED-ray endpoints are conditioning-limited (≈ ε_f32/Δ ~ 1e−4 at the capture radius, cf. V-A2 note) and are reported separately |

Run on every merge touching `src/physics`, `src/shaders`, or integrator parameters; full sweep on release. **Every performance change re-runs V-O** (IMPLEMENTATION.md M6).

## 3. Rendering Validation — Ring 3 (V-I)

Framings ship as presets (single source of truth with the product). Comparisons are tolerance-based, never pixel-exact (D12: cross-driver FMA nondeterminism).

| ID | Test | Reference | Pass criterion |
|---|---|---|---|
| V-I1 | Kerr shadow contour | Bardeen parametric curve [R1], `a ∈ {0, 0.5, 0.9, 0.998}` | Rendered boundary within 1 px of analytic overlay at 1080p reference framing |
| V-I2 | Schwarzschild disk isophotes | Luminet 1979 Fig. 11 [R2] | Primary & secondary image isophote shapes within 2 % of published geometry |
| V-I3 | High-spin disk morphology | DNGR published frames [R5] (beaming ON variants) | Feature-position agreement; beaming asymmetry ratio within 5 % of oracle render |
| V-I4 | Einstein ring radius | Weak-field lens equation for displaced source | Ring angular radius rel error `< 1 %` at `b ≫ b_c` framing |
| V-I5 | Golden regression | Baselined converged renders of all shipped presets | Perceptual diff (SSIM > 0.99) **and** physical diff (mean relative HDR luminance error < 1 %) pre-tonemap |

Comparison overlays are committed to `docs/` at each release — the project's auditable scientific record.

## 4. Performance Validation (V-P)

Reference hardware: RTX 4070 Ti, Windows 11, Chrome stable. Metrics are medians over a fixed scripted camera path (flakiness control).

| ID | Test | Target |
|---|---|---|
| V-P1 | Interactive throughput | ≥ 60 fps at 1080p-equivalent adaptive quality, full disk + sky, scripted orbit |
| V-P2 | Pass budgets | Pass B ≤ 70 % of frame; Passes A+C+D ≤ 3 ms combined (timestamp queries) |
| V-P3 | Still-frame convergence | Converged (V-I5-quality) frame ≤ 2 s after camera rest at 1080p |
| V-P4 | TDR safety | No single dispatch > 250 ms on reference hardware; device-lost recovery exercised in test |

Benchmark plan: `npm run bench` drives the scripted path headed, records per-pass timings from the profiler, writes a JSON report; reports are kept per release for trend tracking. Secondary tier (integrated GPU, e.g. Iris Xe): correctness required (V-O at Low tier), performance reported but not gated.

## 5. Regression Testing

- **CI (every PR):** typecheck → V-A suite → build → V-I5 on SwiftShader WebGPU adapter with relaxed thresholds (software rasterizer differences).
- **Merge to main touching physics/shaders:** V-O harness (requires GPU — run locally or on self-hosted runner; result JSON attached to PR).
- **Release tag:** full V-A + V-O sweep + V-I1–V-I5 on reference hardware + V-P; comparison figures regenerated into `docs/`.
- **Baseline updates:** V-I5 goldens may only be re-baselined in a dedicated PR that states the physical reason for the change.

## 6. Expected Outputs

Qualitative signatures any correct build must show (fast human smoke test):

1. `a = 0`: circular shadow; silhouette diameter `2b_c = 10.39 M` (angular size per framing).
2. `a = 0.998`, edge-on: D-shaped shadow displaced toward retrograde side.
3. Disk edge-on: upper secondary image arcs over the shadow; approaching (left, for standard framing) side dramatically brighter — if the disk looks symmetric, beaming is broken.
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
| CI image tests run on SwiftShader, not real GPU | GitHub-hosted runner limits; real-GPU sweep is a release-gate manual/self-hosted step | §5 |
