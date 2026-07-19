# LensForge — Architecture

System structure, module boundaries, pipelines, and binding design decisions (`D#`). Physics content: PHYSICS.md. Milestones: IMPLEMENTATION.md.

## 1. High-Level Architecture

```
 ┌────────┐   writes    ┌────────────────┐   dirty flags   ┌───────────────┐
 │   UI   │ ──────────▶ │ ParameterStore │ ──────────────▶ │ Frame Graph   │
 └────────┘             │  (src/sim)     │                 │ (src/renderer)│
 ┌────────┐   writes    │  single source │   uniforms      │  A→B*→C→D     │
 │ Camera │ ──────────▶ │  of truth      │ ──────────────▶ │  chunk sched. │
 │ (scene)│             └────────────────┘                 └──────┬────────┘
 └────────┘                                                       │ dispatch
 ┌──────────────────┐        validates                     ┌──────▼────────┐
 │ src/physics f64  │ ◀──────────────────────────────────▶ │ src/gpu + WGSL│
 │ oracle + tests   │     src/validation harness           │ kernels (f32) │
 └──────────────────┘                                      └───────────────┘
```

Two independent implementations of the same physics formulation: a float64 CPU oracle (`src/physics/`) and float32 WGSL kernels (`src/shaders/`). `src/validation/` proves they agree (VALIDATION.md §V-O). All real-time work happens on the GPU; the CPU handles parameters, camera, and precomputation only.

## 2. Module Responsibilities

| Module | Responsibility | Must not |
|---|---|---|
| `src/core/` | Vec/mat/quaternion math, constants, unit conversion (geometrized ↔ SI), shared buffer-layout definitions | Depend on any other module |
| `src/physics/` | Float64 reference: Kerr metric, Hamiltonian RHS, DP45 integrator, tetrads, circular orbits, Novikov–Thorne flux, redshift factor | Touch GPU, DOM, or rendering |
| `src/shaders/` | WGSL kernel sources, mirroring `src/physics/` one-to-one where applicable | Contain physics absent from `src/physics/` |
| `src/gpu/` | Device acquisition, capability tiering, pipeline/bind-group builders, buffer lifecycle, WGSL preprocessor glue, timestamp profiler | Know about physics or frame structure |
| `src/renderer/` | Frame graph (passes A–D), chunk scheduler, render targets, accumulation, tiled export | Compute physics beyond buffer layouts from `core` |
| `src/scene/` | Camera controller, observer tetrad selection, preset serialization (JSON schema) | Own simulation time |
| `src/sim/` | Simulation clock (time scaling), ParameterStore with dirty-flag diffing | Render or own the camera |
| `src/ui/` | Tweakpane bindings → ParameterStore only | Touch GPU or renderer directly |
| `src/validation/` | GPU-vs-oracle harness, golden data, drift reports | Ship in production bundle |
| `src/app/` | Bootstrap, capability page, wiring | Contain logic beyond composition |
| `tools/` | Offline Node scripts: sky-map build, NT flux LUT, Planck→XYZ LUT, bright-star catalog extraction | Run in the browser |

## 3. Folder Structure

```
lensforge/
├─ src/
│  ├─ core/          ├─ physics/       ├─ shaders/
│  │                 │                 │   ├─ common/   (math, RNG jitter)
│  │                 │                 │   ├─ metric/   ├─ integrator/
│  │                 │                 │   ├─ disk/     ├─ sky/  └─ post/
│  ├─ gpu/           ├─ renderer/      ├─ scene/
│  ├─ sim/           ├─ ui/            ├─ validation/   └─ app/
├─ assets/           (KTX2 sky maps, LUTs, star catalog extract)
├─ tools/            (offline precompute scripts)
├─ tests/
│  ├─ unit/          (Vitest, runs src/physics in CI)
│  └─ golden/        (committed V-I5 golden PNGs + baselines.json)
├─ docs/             (release validation record: figures + raw results + README media — prose lives in root .md files)
└─ .github/          (CI + release workflows, issue/PR templates)
```

## 4. Rendering Pipeline

```
Pass A  RAY INIT (compute)      pixel → tetrad direction → (x^μ, p_μ) written to
                                RayState storage buffer (SoA layout)
Pass B  GEODESIC MARCH          ≤ K RK45 steps per dispatch per ray; loops until all
        (compute, CHUNKED)      rays leave MARCHING. Statuses: MARCHING | CAPTURED |
                                ESCAPED | HIT_DISK(r, φ, g). H-drift monitor per step.
Pass C  SHADE (compute)         ESCAPED   → sky cubemap + analytic star splats × |det J|⁻¹
                                HIT_DISK  → Planck(T_eff(r)) × g³ → CIE XYZ → linear sRGB
                                CAPTURED  → 0
                                accumulates into a float32 mean (temporal AA) mirrored to
                                the rgba16f HDR target (presented + 8-bit PNG export source)
Pass D  POST (compute/render)   temporal accumulation → optional physical bloom (off by
                                default) → exposure → AgX tone map → present / PNG export
```

## 5. GPU Pipeline

- **Persistent ray state** in storage buffers, SoA. Rays are a checkpointable process, never "one dispatch = one frame" (D6).
- **Chunk scheduler** (renderer): dispatches Pass B with a per-dispatch step budget `K` sized by measured pass timings; requeues while any ray is MARCHING. The same scheduler drives interactive frames, progressive refinement, and tiled 8K export — one code path.
- **Interaction presentation contract (M5, extended M6):** the presented image is complete for the current pose every frame. A *preview ladder* of the same A–B–C trace runs at descending scales (¼ then ½ resolution; same shaders/physics/uniform words, per-level SoA state and marching counters), coarsest first; each level's Pass C presents the next-coarser level's radiance for its own still-MARCHING pixels (bilinear, exactly aligned — every grid samples the same tan-space frustum), and the full-resolution shade presents the finest level. The full-resolution march ladder also runs on interaction frames, so rays that terminate in-frame present at full resolution immediately — during motion most of the frame is pixel-exact and only the slow geodesic tail falls back down the ladder. No stale frames, no pose mixing; converged output is bit-identical to a preview-less trace.
- **Frame-budget governor (M6):** chunk budgets per level are the baselines × `workScale`, a feedback control on the measured total trace GPU time per frame (timestamp-query EMAs) against a fixed target. The coarsest level's completeness ladder is never scaled. On the dev GPU the mid-drag frame measures pixel-identical to the converged frame at `a ≤ 0.9`; the governor floor keeps `a = 0.998` within ~2 % of pixels at ladder resolution during motion.
- **Temporal accumulation on rest (M6):** after the base trace converges (sample 0 — exact pixel centers, the image all V-O/V-I comparisons use), Halton-jittered refinement rounds re-trace the frame and fold converged rounds into a float32 running mean (32 samples max) — the physically correct pixel-footprint integral. Accumulation shades run exactly once per completed round (re-shading would bias the mean); any parameter change restarts at the base sample.
- **Production sky (M7, D9):** `tools/generateSkyAssets.ts` builds the shipped set from pinned public sources (NASA SVS Deep Star Maps 2020 EXR + Yale BSC5): every bright catalog star is measured photometrically *in* the panorama, subtracted from it, and shipped as an analytic catalog — cubemap and splat fluxes share one unit system by construction, and stars too faint to measure stay baked in (energy conserved, nothing double-counted). Pass C samples the star-subtracted KTX2 cubemap by arrival direction (exact for extended sources) and splats catalog stars with `|det J|⁻¹` magnification, `J` finite-differenced from the 4-neighborhood's escape directions (PHYSICS.md §8); a CSR (θ, φ)-cell grid bounds the per-pixel star scan, and heavily demagnified pixels blend into a 64² aggregate-star cubemap (the unresolved-field limit). The displayed image is the traced image mirrored in screen x with φ-mirrored sky lookup — the exact arriving-photon parity map (PHYSICS.md §8).
- **Tiled export (M7):** the trace uniforms carry tile words (pixel offset + full-image dimensions); `init.wgsl` computes NDC in full-image space, so a tile renders exactly the geodesics of a monolithic render (interactive rendering is tile (0, 0, w, h), bit-identical to the untiled mapping). The exporter (`src/renderer/export.ts`) renders 1024² tiles with a 1-pixel apron (interior tile borders get the same central-difference Jacobians as a full frame), converges each tile plus 8 temporal-AA rounds through the same scheduler, applies the present transform to rgba8, and assembles the PNG (to 8K). Measured: tiled output is bit-identical to the converged monolithic frame. Export mode disables the preview ladder and runs the maximum chunk ladder per frame — every dispatch stays bounded at 64 DP45 attempts, so V-P4 TDR safety is unchanged.
- **March workgroup size** is measured at startup (`src/renderer/marchBenchmark.ts`): a standard ray batch marched at 64/128/256 threads per group, best median wins (vendor-dependent; 64≈128 > 256 on the dev GPU).
- **Adapter-limit compliance:** the device is created with the adapter's full `maxStorageBufferBindingSize`/`maxBufferSize` (the spec defaults are 128 MiB/256 MiB — below one fullscreen ray state at 44 B/pixel), and every trace grid (screen and preview levels) is allocated as the minimum number of horizontal **slabs** whose buffers fit the granted limits and whose 1D march dispatch fits `maxComputeWorkgroupsPerDimension`. Each slab is a tile of the same image through the M7 tile-uniform path with a 1-row apron at interior borders (central-difference Jacobians preserved), shaded into its own target and composed by copy; a grid that fits in one slab reproduces the pre-split allocation and dispatch pattern byte-for-byte. Measured: slab-split output is bit-identical to the monolithic render (0 differing HDR values at 720p ×5 slabs and 4K ×3 slabs); V-I5 goldens hold at SSIM = 1.0 under forced splitting.
- **Compaction decision (D8, M6 evidence):** not implemented. At `a = 0.998` the convergence tail (< 10 % rays marching) dominates wall time, but it is serial-depth-bound (few rays × tens of thousands of steps at 64 steps/chunk), not occupancy-bound — prefix-sum compaction cannot shorten the serial chain; the governor's chunk scaling is the effective mitigation.
- **Specialization constants** for quality tier and `K`; WGSL assembled at build time by a Vite preprocessor plugin (WGSL has no `#include`).
- **Timestamp queries** wrap every pass; feeds the in-app profiler HUD (DEVELOPMENT.md §Profiling).
- Optional prefix-sum ray compaction, added only if profiling shows divergence-tail cost > ~20% (see §8, D8).

## 6. Data Flow

Per frame: UI/camera mutate ParameterStore → diff decides **restart accumulation** (physics parameter changed) vs **continue** (camera at rest; or only time advanced with axisymmetric disk) → single uniform buffer packed (`a`, `Ṁ`, camera pose/tetrad, FOV, `t_sim`, tolerances, `K`) → passes A–D. GPU→CPU readbacks: profiling queries and PNG export only.

**Simulation loop** and **render loop** are decoupled: the sim loop (per rAF) advances `t_sim += dt_real × timeScale` and camera orbit — microseconds of CPU work, no CPU physics stepping. The render loop asks whether the accumulation target matches current uniforms and either refines or restarts. This is what makes still-frame quality unbounded while interaction stays fluid.

## 7. Dependency Graph

```
core ◀── physics ◀── validation, tools, tests/unit
core ◀── gpu ◀── renderer ◀── app
sim, scene ◀── app          ui ──▶ sim (only)
shaders: no build-time deps except the preprocessor
```

Enforced acyclic. Runtime dependency: Tweakpane. Dev dependencies: Vite, Vitest, TypeScript (plus offline tool libraries in `tools/`).

## 8. Design Decisions

| ID | Decision | Rationale (condensed) |
|---|---|---|
| D1 | WebGPU only, no WebGL2 fallback | Compute shaders, storage buffers, timestamp queries are load-bearing; a fragment-shader fallback is a permanent tax. Capability page instead. |
| D2 | No engine (no Three.js) | No meshes, no scene graph — one compute integrator + post chain. An engine is an abstraction to work around. |
| D3 | Hamiltonian geodesics (8 ODEs), not Christoffel or Carter-separated forms | Needs only inverse metric + gradient; no turning-point sign tracking (classic GPU branch bug source); ~2–3× less arithmetic than second-order form. PHYSICS.md §4. |
| D4 | Boyer–Lindquist coordinates | Rays terminate at `r₊(1+ε)` — the BL coordinate singularity is unreachable. Direct comparability with published validation data [R1][R2][R5][R6]. Kerr–Schild is documented fallback if near-horizon step collapse bites. |
| D5 | Kernel units `G = c = M = 1` | Centers arithmetic in float32's precision sweet spot; mass enters only via UI-side unit conversion. PHYSICS.md §1. |
| D6 | Chunked, checkpointable ray march | Windows TDR (~2 s watchdog) resets the device on long dispatches. Also yields progressive refinement and tiled export for free. |
| D7 | No WASM/Rust in core | Physics hot loop is on the GPU; CPU load is trivial. The float64 oracle in TypeScript suffices for validation. |
| D8 | Ray retirement over eager compaction | Early-terminating rays stop consuming work via status masks; prefix-sum compaction only if profiling justifies it. |
| D9 | Hybrid sky: HDR cubemap + analytic bright-star splats scaled by `|det J|⁻¹` | Surface brightness conservation makes cubemap lookup automatically correct; point sources need explicit magnification — texture resolution can never fix star aliasing under strong distortion. |
| D10 | HDR mandatory, AgX tone map | Beaming produces 10³–10⁴ radiance ratios; AgX over ACES for hue behavior under extreme ratios. |
| D11 | Single quality knob = integrator tolerance | All quality/perf trade-offs route through one parameter plus dynamic resolution; no scattered magic constants. |
| D12 | Image tests are tolerance-based, never pixel-exact | WGSL FMA contraction differs across drivers; exact match is impossible by design. Exact tests live in the CPU oracle suite. |

## 9. Future Scalability

The architecture anticipates (without implementing) the expansions in this order of value/effort: volumetric disk as an emissive ray-marched volume fed by imported GRMHD snapshots (slots into Pass B as an alternative medium model); free-fall camera on timelike geodesics (same integrator, `H = −½`); df64 double-float offline mode for n ≥ 2 photon subrings (integrator is formulation-agnostic over the scalar type); relativistic jets (additional emitter model in Pass C); spectral rendering replacing the Planck→XYZ LUT; WebGPU subgroups for integrator throughput. None of these require changing module boundaries — only adding implementations behind existing seams.
