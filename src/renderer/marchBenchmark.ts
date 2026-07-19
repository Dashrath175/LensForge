/**
 * Startup micro-benchmark (IMPLEMENTATION.md M6): pick the geodesic-march
 * workgroup size for this device. The march kernel is occupancy-bound and
 * divergence-heavy; the best size varies across vendors, so it is measured,
 * not assumed. A standard batch of equatorial rays spanning the deflection
 * range is marched for a fixed chunk count at each candidate size; wall time
 * around queue-completion ranks them (relative ranking only — absolute pass
 * timing stays with the timestamp profiler).
 */
import type { GpuContext } from '../gpu/device';
import { GEODESIC_DEFAULTS } from '../physics/integrator';
import { horizonRadius } from '../physics/metric';
import { equatorialPhoton } from '../physics/photon';
import marchWgsl from '../shaders/trace/march.wgsl';

export const MARCH_WORKGROUP_CANDIDATES = [64, 128, 256] as const;

const BENCH_RAYS = 65536;
const BENCH_CHUNKS = 6;
const BENCH_SPIN = 0.9;
const STATE_PLANES = 11;
const UNIFORM_BYTES = 224; // mirrors tracePass.ts

export interface MarchBenchmarkResult {
  workgroupSize: number;
  /** Median wall ms per candidate, for the HUD/diagnostics. */
  timingsMs: Record<number, number>;
}

export async function benchmarkMarchWorkgroupSize(gpu: GpuContext): Promise<MarchBenchmarkResult> {
  const { device } = gpu;

  // Standard ray population: impact parameters from near-critical to weak
  // deflection (the real workload mix), all inbound from r = 100.
  const states = new Float32Array(STATE_PLANES * BENCH_RAYS);
  for (let i = 0; i < BENCH_RAYS; i++) {
    const b = 4 + (8 * i) / BENCH_RAYS;
    const y = equatorialPhoton(BENCH_SPIN, 100, b, true);
    for (let p = 0; p < 8; p++) states[p * BENCH_RAYS + i] = y[p]!;
    states[9 * BENCH_RAYS + i] = 1e-4; // errOld
  }

  const uniformData = new ArrayBuffer(UNIFORM_BYTES);
  const f32 = new Float32Array(uniformData);
  const u32 = new Uint32Array(uniformData);
  f32[23] = BENCH_SPIN;
  u32[34] = BENCH_RAYS;
  u32[35] = 64; // steps per chunk, as in production
  f32.set(
    [1e-6, 1e-8, horizonRadius(BENCH_SPIN) * (1 + GEODESIC_DEFAULTS.horizonEpsilon), GEODESIC_DEFAULTS.rEscape],
    36,
  );
  f32.set([0, 0, 1, 0], 48);
  const uniforms = device.createBuffer({
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'bench-uniforms',
  });
  device.queue.writeBuffer(uniforms, 0, uniformData);

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const state = device.createBuffer({ size: states.byteLength, usage, label: 'bench-state' });
  const status = device.createBuffer({ size: BENCH_RAYS * 4, usage, label: 'bench-status' });
  const result = device.createBuffer({ size: BENCH_RAYS * 3 * 4, usage, label: 'bench-result' });
  const counter = device.createBuffer({ size: 4, usage, label: 'bench-counter' });
  const module = device.createShaderModule({ code: marchWgsl, label: 'bench-march' });

  const timingsMs: Record<number, number> = {};
  let best = 64;
  let bestMs = Infinity;
  try {
    for (const size of MARCH_WORKGROUP_CANDIDATES) {
      const pipeline = device.createComputePipeline({
        label: `bench-march-${size}`,
        layout: 'auto',
        compute: { module, entryPoint: 'cs_march', constants: { WORKGROUP_SIZE: size } },
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniforms } },
          { binding: 1, resource: { buffer: state } },
          { binding: 2, resource: { buffer: status } },
          { binding: 3, resource: { buffer: result } },
          { binding: 4, resource: { buffer: counter } },
        ],
      });

      const runs: number[] = [];
      for (let rep = 0; rep < 3; rep++) {
        // Fresh rays each rep, so every candidate marches identical work.
        device.queue.writeBuffer(state, 0, states);
        device.queue.writeBuffer(status, 0, new Uint32Array(BENCH_RAYS));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass({ label: `bench-${size}` });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        for (let c = 0; c < BENCH_CHUNKS; c++) {
          pass.dispatchWorkgroups(Math.ceil(BENCH_RAYS / size));
        }
        pass.end();
        const t0 = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        runs.push(performance.now() - t0);
      }
      runs.sort((a, b) => a - b);
      const median = runs[1]!; // rep 0 doubles as warmup; median damps outliers
      timingsMs[size] = median;
      if (median < bestMs) {
        bestMs = median;
        best = size;
      }
    }
  } finally {
    uniforms.destroy();
    state.destroy();
    status.destroy();
    result.destroy();
    counter.destroy();
  }
  return { workgroupSize: best, timingsMs };
}
