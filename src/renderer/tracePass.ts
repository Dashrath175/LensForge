/**
 * Passes A-C: ray init, chunked geodesic march, shade (ARCHITECTURE.md §4-§5).
 *
 * Owns the persistent SoA ray-state buffers and the chunk scheduling contract
 * (D6): every march dispatch is bounded by stepsPerDispatch DP45 attempts, and
 * an atomic still-marching counter (read back asynchronously) tells the
 * scheduler when the image has converged. Camera-frame quantities are computed
 * in float64 via src/physics and uploaded as uniforms.
 *
 * `runBatch` drives the same march pipeline over caller-supplied rays; it is
 * the GPU side of the V-O validation harness (src/validation).
 */
import type { GpuContext } from '../gpu/device';
import type { GpuProfiler } from '../gpu/profiler';
import { GEODESIC_DEFAULTS } from '../physics/integrator';
import { createMetric, evalMetric, horizonRadius } from '../physics/metric';
import { orthonormalTetrad, zamoFourVelocity } from '../physics/tetrad';
import initWgsl from '../shaders/trace/init.wgsl';
import marchWgsl from '../shaders/trace/march.wgsl';
import shadeWgsl from '../shaders/trace/shade.wgsl';

// Mirrors src/shaders/common/ray-state.wgsl — change both together.
export const STATUS_MARCHING = 0;
export const STATUS_CAPTURED = 1;
export const STATUS_ESCAPED = 2;
export const STATUS_MAX_STEPS = 3;
const STATE_PLANES = 11;
const RESULT_PLANES = 3;

const UNIFORM_BYTES = 160;
const WORKGROUP_1D = 64;
const WORKGROUP_2D = 8;

/**
 * The single quality knob (D11): integrator tolerance per tier, tuned against
 * the V-O1 bounds (1/3/10 arcmin). Escape-direction error scales as
 * ~steps × rtol × |y|, so High tier sits near the f32 truncation floor.
 */
export const QUALITY_TIERS = {
  high: { rtol: 1e-6, atol: 1e-8 },
  medium: { rtol: 1e-5, atol: 1e-7 },
  low: { rtol: 1e-4, atol: 1e-6 },
} as const;
export type QualityTier = keyof typeof QUALITY_TIERS;

/** DP45 attempts per ray per dispatch; bounds dispatch duration (V-P4). */
const STEPS_PER_DISPATCH = 64;
/** March dispatches encoded per frame while unconverged (1 = best frame
 * pacing during interaction; M6 makes this budget-adaptive). */
const DISPATCHES_PER_FRAME = 1;
/** Angular radius of the antipodal sky marker (Einstein-ring source), radians. */
const MARKER_OUTER_RAD = 0.02;
const MARKER_INNER_RAD = 0.012;

export interface CameraPose {
  r: number;
  theta: number;
  phi: number;
  fovY: number;
}

export interface BatchResult {
  statuses: Uint32Array;
  /** Final geodesic states, 8 components per ray (f32 lifted to f64). */
  states: Float64Array;
  /** Escape directions, 3 per ray (valid where status is ESCAPED). */
  directions: Float64Array;
  steps: Float64Array;
  dispatches: number;
}

interface RayBuffers {
  rayCount: number;
  state: GPUBuffer;
  status: GPUBuffer;
  result: GPUBuffer;
}

export class TracePass {
  private readonly gpu: GpuContext;
  private readonly profiler: GpuProfiler;

  private readonly initPipeline: GPUComputePipeline;
  private readonly marchPipeline: GPUComputePipeline;
  private readonly shadePipeline: GPUComputePipeline;

  private readonly uniformBuffer: GPUBuffer;
  private readonly uniformData = new ArrayBuffer(UNIFORM_BYTES);
  private readonly counter: GPUBuffer;
  private readonly counterStaging: GPUBuffer;
  private counterMapPending = false;

  private screen: RayBuffers | null = null;
  private initBindGroup!: GPUBindGroup;
  private marchBindGroup!: GPUBindGroup;
  private shadeBindGroup!: GPUBindGroup;

  private width = 0;
  private height = 0;
  private camera: CameraPose | null = null;
  private spin = 0;
  private tier: QualityTier = 'high';
  private needsInit = false;
  private knownMarching = 0;
  /** Bumped on every camera change; stale counter readbacks (initiated under
   * an older generation) are discarded, or they could zero knownMarching
   * right after a re-init and stall the march with an unconverged image. */
  private generation = 0;

  constructor(gpu: GpuContext, profiler: GpuProfiler, hdrView: GPUTextureView, width: number, height: number) {
    this.gpu = gpu;
    this.profiler = profiler;
    const { device } = gpu;

    const make = (code: string, entryPoint: string, label: string): GPUComputePipeline =>
      device.createComputePipeline({
        label,
        layout: 'auto',
        compute: { module: device.createShaderModule({ code, label }), entryPoint },
      });
    this.initPipeline = make(initWgsl, 'cs_init', 'trace-init');
    this.marchPipeline = make(marchWgsl, 'cs_march', 'trace-march');
    this.shadePipeline = make(shadeWgsl, 'cs_shade', 'trace-shade');

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'trace-uniforms',
    });
    this.counter = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'marching-counter',
    });
    this.counterStaging = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'marching-counter-staging',
    });

    this.resize(width, height, hdrView);
  }

  private createRayBuffers(rayCount: number, label: string): RayBuffers {
    const { device } = this.gpu;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return {
      rayCount,
      state: device.createBuffer({ size: STATE_PLANES * rayCount * 4, usage, label: `${label}-state` }),
      status: device.createBuffer({ size: rayCount * 4, usage, label: `${label}-status` }),
      result: device.createBuffer({ size: RESULT_PLANES * rayCount * 4, usage, label: `${label}-result` }),
    };
  }

  private destroyRayBuffers(b: RayBuffers): void {
    b.state.destroy();
    b.status.destroy();
    b.result.destroy();
  }

  resize(width: number, height: number, hdrView: GPUTextureView): void {
    this.width = width;
    this.height = height;
    if (this.screen) this.destroyRayBuffers(this.screen);
    this.screen = this.createRayBuffers(width * height, 'screen');

    const { device } = this.gpu;
    const uniforms = { binding: 0, resource: { buffer: this.uniformBuffer } };
    this.initBindGroup = device.createBindGroup({
      layout: this.initPipeline.getBindGroupLayout(0),
      entries: [
        uniforms,
        { binding: 1, resource: { buffer: this.screen.state } },
        { binding: 2, resource: { buffer: this.screen.status } },
      ],
    });
    this.marchBindGroup = device.createBindGroup({
      layout: this.marchPipeline.getBindGroupLayout(0),
      entries: [
        uniforms,
        { binding: 1, resource: { buffer: this.screen.state } },
        { binding: 2, resource: { buffer: this.screen.status } },
        { binding: 3, resource: { buffer: this.screen.result } },
        { binding: 4, resource: { buffer: this.counter } },
      ],
    });
    this.shadeBindGroup = device.createBindGroup({
      layout: this.shadePipeline.getBindGroupLayout(0),
      entries: [
        uniforms,
        { binding: 2, resource: { buffer: this.screen.status } },
        { binding: 3, resource: { buffer: this.screen.result } },
        { binding: 5, resource: hdrView },
      ],
    });

    if (this.camera) this.setCamera(this.camera, this.spin);
  }

  /** Compute the observer frame in float64 (src/physics) and upload uniforms. */
  setCamera(pose: CameraPose, spin = 0): void {
    this.camera = pose;
    this.spin = spin;

    const m = createMetric();
    evalMetric(spin, pose.r, pose.theta, m);
    const tetrad = orthonormalTetrad(m, zamoFourVelocity(m, spin, pose.r));

    const tanHalfFovY = Math.tan(pose.fovY / 2);
    const tanHalfFovX = (tanHalfFovY * this.width) / this.height;
    const sinTh = Math.sin(pose.theta);
    const marker = [
      -sinTh * Math.cos(pose.phi),
      -sinTh * Math.sin(pose.phi),
      -Math.cos(pose.theta),
    ];
    const tolerance = QUALITY_TIERS[this.tier];

    const f32 = new Float32Array(this.uniformData);
    const u32 = new Uint32Array(this.uniformData);
    for (let leg = 0; leg < 4; leg++) f32.set(tetrad[leg]!, leg * 4);
    f32.set([m.g_tt, m.g_tph, m.g_rr, m.g_thth], 16);
    f32.set([m.g_phph, tanHalfFovX, tanHalfFovY, spin], 20);
    f32.set([pose.r, pose.theta, pose.phi, Math.cos(MARKER_INNER_RAD)], 24);
    f32.set([marker[0]!, marker[1]!, marker[2]!, Math.cos(MARKER_OUTER_RAD)], 28);
    u32.set([this.width, this.height, this.width * this.height, STEPS_PER_DISPATCH], 32);
    f32.set(
      [
        tolerance.rtol,
        tolerance.atol,
        horizonRadius(spin) * (1 + GEODESIC_DEFAULTS.horizonEpsilon),
        GEODESIC_DEFAULTS.rEscape,
      ],
      36,
    );
    this.gpu.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

    this.needsInit = true;
    this.knownMarching = this.width * this.height;
    this.generation++;
  }

  setQuality(tier: QualityTier): void {
    this.tier = tier;
    if (this.camera) this.setCamera(this.camera, this.spin);
  }

  getCamera(): CameraPose | null {
    return this.camera;
  }

  getKnownMarching(): number {
    return this.camera ? this.knownMarching : 0;
  }

  /** Encode passes A-C for this frame. Returns false when no camera is set. */
  encode(encoder: GPUCommandEncoder): boolean {
    if (!this.camera || !this.screen) return false;
    const groupsX = Math.ceil(this.width / WORKGROUP_2D);
    const groupsY = Math.ceil(this.height / WORKGROUP_2D);

    if (this.needsInit) {
      const pass = encoder.beginComputePass({
        label: 'trace-init',
        timestampWrites: this.profiler.timestampWrites('init') as GPUComputePassTimestampWrites,
      });
      pass.setPipeline(this.initPipeline);
      pass.setBindGroup(0, this.initBindGroup);
      pass.dispatchWorkgroups(groupsX, groupsY);
      pass.end();
      this.needsInit = false;
      this.knownMarching = this.screen.rayCount;
    }

    if (this.knownMarching > 0) {
      for (let d = 0; d < DISPATCHES_PER_FRAME; d++) {
        encoder.clearBuffer(this.counter);
        const pass = encoder.beginComputePass({
          label: 'trace-march',
          timestampWrites: this.profiler.timestampWrites(`march${d}`) as GPUComputePassTimestampWrites,
        });
        pass.setPipeline(this.marchPipeline);
        pass.setBindGroup(0, this.marchBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.screen.rayCount / WORKGROUP_1D));
        pass.end();
      }
      if (!this.counterMapPending) {
        encoder.copyBufferToBuffer(this.counter, 0, this.counterStaging, 0, 4);
      }
    }

    const pass = encoder.beginComputePass({
      label: 'trace-shade',
      timestampWrites: this.profiler.timestampWrites('shade') as GPUComputePassTimestampWrites,
    });
    pass.setPipeline(this.shadePipeline);
    pass.setBindGroup(0, this.shadeBindGroup);
    pass.dispatchWorkgroups(groupsX, groupsY);
    pass.end();
    return true;
  }

  /** Kick off the async still-marching counter readback; call once per frame after submit. */
  afterSubmit(): void {
    if (this.counterMapPending || this.knownMarching === 0) return;
    this.counterMapPending = true;
    const generationAtRead = this.generation;
    void this.counterStaging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const count = new Uint32Array(this.counterStaging.getMappedRange())[0]!;
        this.counterStaging.unmap();
        if (generationAtRead === this.generation) this.knownMarching = count;
      })
      .catch(() => {
        /* device loss: recovery rebuilds this pass */
      })
      .finally(() => {
        this.counterMapPending = false;
      });
  }

  /** Read back the screen status plane (validation: shadow-diameter measurement). */
  async readScreenStatuses(): Promise<Uint32Array> {
    if (!this.screen) throw new Error('trace pass has no screen buffers');
    return new Uint32Array((await this.readBuffer(this.screen.status)).buffer);
  }

  private async readBuffer(src: GPUBuffer): Promise<Uint8Array> {
    const { device } = this.gpu;
    const staging = device.createBuffer({
      size: src.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'readback-staging',
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, staging, 0, src.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return bytes;
  }

  /**
   * March caller-supplied rays (8 f64 components each) to termination on the
   * GPU with the current tier's tolerances. The GPU integrates the f32-rounded
   * states — the honest end-to-end comparison for V-O (VALIDATION.md §2).
   */
  async runBatch(initialStates: Float64Array, spin: number, maxDispatches = 400): Promise<BatchResult> {
    const rayCount = initialStates.length / 8;
    const { device } = this.gpu;
    const buffers = this.createRayBuffers(rayCount, 'batch');

    // SoA planes from the AoS input; h = 0 sentinel, errOld = 1e-4, steps = 0.
    const planes = new Float32Array(STATE_PLANES * rayCount);
    for (let i = 0; i < rayCount; i++) {
      for (let p = 0; p < 8; p++) planes[p * rayCount + i] = initialStates[i * 8 + p]!;
      planes[9 * rayCount + i] = 1e-4;
    }
    device.queue.writeBuffer(buffers.state, 0, planes);
    device.queue.writeBuffer(buffers.status, 0, new Uint32Array(rayCount)); // all MARCHING

    const tolerance = QUALITY_TIERS[this.tier];
    const uniformData = this.uniformData.slice(0);
    const f32 = new Float32Array(uniformData);
    const u32 = new Uint32Array(uniformData);
    f32[23] = spin; // camMetricB.w
    u32[34] = rayCount;
    u32[35] = STEPS_PER_DISPATCH;
    f32.set(
      [
        tolerance.rtol,
        tolerance.atol,
        horizonRadius(spin) * (1 + GEODESIC_DEFAULTS.horizonEpsilon),
        GEODESIC_DEFAULTS.rEscape,
      ],
      36,
    );
    const batchUniforms = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'batch-uniforms',
    });
    device.queue.writeBuffer(batchUniforms, 0, uniformData);

    const bindGroup = device.createBindGroup({
      layout: this.marchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: batchUniforms } },
        { binding: 1, resource: { buffer: buffers.state } },
        { binding: 2, resource: { buffer: buffers.status } },
        { binding: 3, resource: { buffer: buffers.result } },
        { binding: 4, resource: { buffer: this.counter } },
      ],
    });
    const counterRead = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'batch-counter-staging',
    });

    let dispatches = 0;
    for (; dispatches < maxDispatches; dispatches++) {
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(this.counter);
      const pass = encoder.beginComputePass({ label: 'batch-march' });
      pass.setPipeline(this.marchPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(rayCount / WORKGROUP_1D));
      pass.end();
      encoder.copyBufferToBuffer(this.counter, 0, counterRead, 0, 4);
      device.queue.submit([encoder.finish()]);
      await counterRead.mapAsync(GPUMapMode.READ);
      const marching = new Uint32Array(counterRead.getMappedRange())[0]!;
      counterRead.unmap();
      if (marching === 0) {
        dispatches++;
        break;
      }
    }

    const stateBack = new Float32Array((await this.readBuffer(buffers.state)).buffer);
    const statusBack = new Uint32Array((await this.readBuffer(buffers.status)).buffer);
    const resultBack = new Float32Array((await this.readBuffer(buffers.result)).buffer);

    const states = new Float64Array(rayCount * 8);
    const directions = new Float64Array(rayCount * 3);
    const steps = new Float64Array(rayCount);
    for (let i = 0; i < rayCount; i++) {
      for (let p = 0; p < 8; p++) states[i * 8 + p] = stateBack[p * rayCount + i]!;
      for (let p = 0; p < 3; p++) directions[i * 3 + p] = resultBack[p * rayCount + i]!;
      steps[i] = stateBack[10 * rayCount + i]!;
    }

    this.destroyRayBuffers(buffers);
    batchUniforms.destroy();
    counterRead.destroy();
    return { statuses: statusBack, states, directions, steps, dispatches };
  }

  destroy(): void {
    if (this.screen) this.destroyRayBuffers(this.screen);
    this.uniformBuffer.destroy();
    this.counter.destroy();
    this.counterStaging.destroy();
  }
}
