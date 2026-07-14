# LensForge

A physically rigorous, real-time black hole renderer running entirely in the browser on WebGPU.

Every visible phenomenon — event horizon shadow, gravitational lensing, Einstein rings, photon ring, Doppler beaming, gravitational redshift, frame dragging — emerges from numerical integration of null geodesics in the Kerr spacetime and exact relativistic radiative transfer. Nothing is faked, painted, or approximated for artistic effect. Where an approximation exists, it is documented.

## Vision

Become one of the highest-quality browser-based black hole simulations ever built: scientific correctness first, GPU performance second, and zero handcrafted visual tricks. See [MASTER_PLAN.md](MASTER_PLAN.md) for the full philosophy, scope, and success criteria.

## Repository Overview

| Path | Contents |
|---|---|
| `src/core/` | Math primitives, constants, unit conversion |
| `src/physics/` | CPU float64 reference physics (the scientific oracle) |
| `src/shaders/` | WGSL kernels (float32 ports of `src/physics/`) |
| `src/gpu/` | WebGPU device layer, pipelines, buffers, profiler |
| `src/renderer/` | Frame graph, passes, chunk scheduler, export |
| `src/scene/` | Camera, observer tetrads, presets |
| `src/sim/` | Simulation clock, parameter store |
| `src/ui/` | Parameter panel bindings |
| `src/validation/` | GPU-vs-oracle comparison harness, golden data |
| `tools/` | Offline asset/LUT generation scripts |
| `tests/` | Unit tests (physics) and golden image tests |
| `assets/` | Sky maps, LUTs, star catalog extract |

Structure rationale: [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick Start

Requirements: Node ≥ 20, a WebGPU-capable browser (Chrome/Edge ≥ 113, Firefox ≥ 141, Safari ≥ 26).

```bash
npm install
npm run dev        # Vite dev server
npm test           # CPU physics test suite (Vitest)
npm run build      # Production build (static output)
```

Full build/debug/deploy workflow: [DEVELOPMENT.md](DEVELOPMENT.md).

## Document Map

| Document | Read it when you need… |
|---|---|
| [MASTER_PLAN.md](MASTER_PLAN.md) | Scope, goals, non-goals, principles, constraints — the source of truth for *what* and *why* |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module boundaries, pipelines, data flow, design decisions (`D#`) |
| [PHYSICS.md](PHYSICS.md) | Every equation, unit convention, numerical method, accuracy limit |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Milestone roadmap (`M0`–`M8`) with exit criteria |
| [VALIDATION.md](VALIDATION.md) | Test definitions (`V-*`), tolerances, benchmarks, expected outputs |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Coding standards, git workflow, debugging, profiling, deployment |
| [REFERENCES.md](REFERENCES.md) | All literature citations (`[R#]`) used across the documentation |

Convention used throughout the docs: `[R#]` cites REFERENCES.md, `D#` cites a design decision in ARCHITECTURE.md, `V-*` cites a test in VALIDATION.md, `M#` cites a milestone in IMPLEMENTATION.md. Information lives in exactly one document; everything else references it.
