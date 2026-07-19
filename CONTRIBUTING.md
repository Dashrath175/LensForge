# Contributing to LensForge

Thank you for your interest. LensForge is a scientific renderer first and a
graphics project second — contributions are welcome, and they are held to the
standards that make the images trustworthy.

Before anything else, read [MASTER_PLAN.md](MASTER_PLAN.md). It is the source
of truth for scope, goals, and non-goals; a change that conflicts with it must
first amend it in its own reviewed PR.

## Ground rules

The full contribution rules live in
[DEVELOPMENT.md §8](DEVELOPMENT.md#8-contribution-rules). The short version:

1. **Physics first.** A physics change lands in `src/physics/` (float64, pure
   functions, unit-tested) in the same PR as — or before — its WGSL port.
   Never WGSL first. No physics may exist in a shader that is absent from
   `src/physics/`.
2. **Approximations are contracts.** Anything that changes accuracy updates
   [PHYSICS.md](PHYSICS.md) §9/§10 and [VALIDATION.md](VALIDATION.md) §7 in
   the same PR.
3. **No AI in the render path.** No learned upscaling, denoising, or
   generation (MASTER_PLAN.md §1.5). Authoring assistance is fine; unverified
   equations are not.
4. **Dependencies are liabilities.** A new runtime dependency needs written
   justification in the PR. The current count is one (Tweakpane).
5. **Golden images** may only be re-baselined in a dedicated PR stating the
   physical cause (VALIDATION.md §5).

## Getting started

```bash
git clone <repo-url>
cd lensforge
npm install
npm run dev        # http://localhost:5173 — needs a WebGPU browser
npm test           # CPU physics suite (no GPU required)
npm run typecheck
```

`npm test` runs the full analytic validation ring (V-A) against closed-form
general relativity — it must pass before and after your change.

## Making a change

- **Branch** from `main`: `type/short-slug` (e.g. `fix/polar-axis-reflection`).
- **Commits:** imperative subject ≤ 72 chars; body explains *why*; reference
  milestone/test IDs where relevant (`M4: fix disk-hit root polish (V-O2)`).
- **CI must be green:** typecheck → V-A suite → build.
- **PRs touching `src/physics/`, `src/shaders/`, or integrator parameters**
  additionally attach a V-O harness report measured on a real GPU: run the
  app with `npm run dev`, then in the browser console
  `await __lensforge.runValidation(0.998)` (see
  [docs/README.md](docs/README.md) for the full command list) and paste the
  JSON into the PR.
- **Performance PRs** include before/after profiler numbers **and** a V-O
  re-run — performance work must not move physics (VALIDATION.md §2).

## What to work on

- Open issues, especially those labeled `good first issue`.
- The future-work seams in [ARCHITECTURE.md §9](ARCHITECTURE.md#9-future-scalability)
  (df64 offline mode, free-fall cameras, spectral rendering, GRMHD volume
  import) are designed extension points — open an issue to discuss scope
  before starting one.
- Documentation and validation tooling improvements are always in scope.

## Reporting a physics discrepancy

If you believe a rendered image is physically wrong, please use the
*Physics discrepancy* issue template. The most useful report contains: the
preset or full parameter state (Presets → Save JSON), what the image shows,
what you expected, and a reference (paper, equation, or independent
calculation). The float64 oracle (`src/physics/`) is the arbiter: a
discrepancy confirmed against the oracle is a bug; a discrepancy *of* the
oracle against published results is a serious bug.

## Code style

TypeScript strict, no `any`; physics code is pure functions documenting their
PHYSICS.md section; WGSL files state their `src/physics` counterpart. Naming
and layout conventions: [DEVELOPMENT.md §1–§2](DEVELOPMENT.md#1-coding-standards).

## Questions

Open a GitHub Discussion (or an issue if Discussions are not enabled). For
conduct concerns, see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
