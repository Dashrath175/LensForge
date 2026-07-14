# LensForge — Development Guide

Workflow, standards, and tooling. Architecture rules: ARCHITECTURE.md. What to build next: IMPLEMENTATION.md.

## 1. Coding Standards

- **TypeScript:** `strict: true`, no `any` (use `unknown` + narrowing), no non-null assertions outside `src/gpu` device bootstrap. ESLint + Prettier enforced in CI; formatting is never discussed in review.
- **Physics code (`src/physics`):** pure functions only — no classes with mutable state, no I/O. Every public function documents its PHYSICS.md section (`/** PHYSICS.md §6 — circular orbit kinematics */`). Variable names match the docs' notation (`Sigma`, `Delta`, `p_t`, `u_t`) even where longer names would be "cleaner" — cross-referencing equations beats naming style.
- **WGSL (`src/shaders`):** each file states its `src/physics` counterpart in a header comment. No physics may exist in WGSL that is absent from `src/physics` (MASTER_PLAN.md §7). No magic numbers — constants come from the shared layout definitions in `src/core` or specialization constants.
- **Comments:** state constraints the code can't express (units, valid domains, why a formulation was chosen → cite `D#`). Never narrate what the next line does.
- **Buffer layouts** are defined once in `src/core` and consumed by both TS and WGSL (generated struct comments); a layout change is a single-file diff.

## 2. Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| TS files | camelCase | `geodesicIntegrator.ts` |
| WGSL files | kebab-case, dir = pipeline stage | `integrator/dp45-step.wgsl` |
| Types/classes | PascalCase | `RayStateBuffer` |
| Constants | SCREAMING_SNAKE | `R_ESCAPE` |
| Tests | mirror source path | `tests/unit/physics/geodesicIntegrator.test.ts` |
| Presets | kebab-case JSON | `dngr-gargantua.json` |
| Branches | `type/short-slug` | `feat/disk-intersection`, `fix/polar-axis-reflection` |

## 3. Repository Workflow & Git Strategy

- **Trunk-based:** `main` is always releasable; short-lived branches (< ~3 days of work), squash-merged via PR. No long-running develop branch.
- **Every PR:** CI green (typecheck, V-A suite, build, SwiftShader V-I5). PRs touching `src/physics`, `src/shaders`, or integrator parameters additionally attach the V-O harness report (VALIDATION.md §5).
- **Physics-first rule (MASTER_PLAN.md §7):** a physics change lands in `src/physics` + unit test in the same PR as (or before) its WGSL port — never WGSL first.
- **Releases:** annotated tag `vX.Y.Z` on `main` after the release-gate sweep (VALIDATION.md §5); tag triggers production deploy.
- **Commits:** imperative subject ≤ 72 chars; body explains *why*; reference milestone/test IDs (`M4: implement disk intersection root-finding (V-O2)`).

## 4. Build Process

```bash
npm run dev        # Vite dev server, WGSL hot reload via preprocessor plugin
npm test           # Vitest: V-A suite + unit tests (CPU only, no GPU needed)
npm run build      # Production: hashed assets, KTX2 sky variants, static output
npm run validate   # V-O harness against local GPU, writes JSON report
npm run golden     # Playwright V-I5 against local GPU
npm run bench      # V-P scripted-path benchmark, writes JSON report
npm run tools:*    # Offline asset/LUT generation (see tools/README header comments)
```

Assets in `assets/` are build inputs, regenerated only by `tools/` scripts — never hand-edited. Pin tool versions inside `tools/`.

## 5. Debugging Workflow

Ordered from cheap to deep:

1. **Physics doubt?** Reproduce the ray in the float64 oracle first (`src/validation` single-ray CLI). If the oracle is wrong, fix physics; if only the GPU differs, it's a port/precision bug — check V-O curves before reading shader code.
2. **In-app profiler HUD** (timestamp queries): per-pass timings, ray-status histograms, MAX_STEPS counter (a rising flag count is the earliest signal of integrator trouble).
3. **Dev-build invariants:** uniform-hash vs accumulation-tag check (stale accumulation), `H`-drift heatmap debug view, ray-status debug view (CAPTURED/ESCAPED/HIT_DISK as colors).
4. **Chrome:** `about:gpu` for adapter state; DevTools performance panel for CPU-side stalls; Dawn validation toggles via `chrome://flags/#enable-webgpu-developer-features` during development.
5. **Deep GPU dives:** PIX for Windows capturing Chrome (launch with `--disable-gpu-sandbox --disable-direct-composition`); use sparingly.
6. **Device lost / TDR:** if reproducible, first suspect a dispatch exceeding the V-P4 bound — check the chunk budget `K` and the exhaustion counter before blaming drivers.

## 6. Performance Profiling

- Ground truth is the in-app timestamp-query profiler, not DevTools (which measures CPU-side submission, not GPU execution).
- Profile before optimizing; the only sanctioned optimization order is ARCHITECTURE.md §5/§8 (retirement → tolerance tiers → dynamic resolution → compaction).
- Any optimization PR includes before/after `npm run bench` reports **and** a V-O re-run (perf must not move physics — VALIDATION.md §2).

## 7. Deployment Process

- **Hosting:** Vercel, static output, zero serverless functions. Nothing in the build is Vercel-specific.
- **Previews:** every PR gets a preview deploy — physics changes are visually reviewable in the PR.
- **Production:** deploy on release tag only, after the release-gate sweep (VALIDATION.md §5).
- **Caching:** immutable-hashed assets, long-cache headers on KTX2 sky maps; capability tier selects 4k vs 8k sky variant at runtime.
- **CI (GitHub Actions):** PR pipeline per §3; release workflow additionally regenerates `docs/` comparison figures and attaches bench/validation reports to the GitHub release.

## 8. Contribution Rules

1. Scope is governed by MASTER_PLAN.md — a PR that conflicts with it must first amend MASTER_PLAN.md in its own reviewed PR.
2. New runtime dependency ⇒ written justification in the PR (MASTER_PLAN.md §7).
3. New approximation or accuracy change ⇒ update PHYSICS.md §9/§10 and VALIDATION.md §7 in the same PR.
4. New design decision ⇒ add a `D#` row to ARCHITECTURE.md §8; reversing one ⇒ edit the row with rationale, don't delete history (git has it).
5. Golden re-baselines only in dedicated PRs with stated physical cause (VALIDATION.md §5).
6. No AI-generated content in the render path, shaders, or physics without full human derivation review against PHYSICS.md — the no-AI-rendering rule (MASTER_PLAN.md §1.5) applies to runtime behavior; authorship assistance is fine, unverified equations are not.
