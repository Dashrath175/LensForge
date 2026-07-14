/**
 * M0 renderer skeleton (ARCHITECTURE.md §4): HDR storage target -> present.
 * The test-pattern compute pass stands where the trace/shade passes (A-C)
 * land in M2; the present pass is the permanent tail of Pass D.
 */
import type { GpuContext } from '../gpu/device';
import { GpuProfiler } from '../gpu/profiler';
import presentWgsl from '../shaders/post/present.wgsl';
import hdrGradientWgsl from '../shaders/test/hdr-gradient.wgsl';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const WORKGROUP_SIZE = 8;

export class Renderer {
  private readonly gpu: GpuContext;
  readonly profiler: GpuProfiler;

  private readonly patternPipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly presentUniforms: GPUBuffer;

  private hdrTexture: GPUTexture;
  private hdrView: GPUTextureView;
  private patternBindGroup!: GPUBindGroup;
  private presentBindGroup!: GPUBindGroup;
  private hdrStale = true;

  constructor(gpu: GpuContext, width: number, height: number) {
    this.gpu = gpu;
    this.profiler = new GpuProfiler(gpu.device, gpu.hasTimestampQuery);
    const { device } = gpu;

    this.patternPipeline = device.createComputePipeline({
      label: 'test-pattern',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: hdrGradientWgsl, label: 'hdr-gradient' }),
        entryPoint: 'cs_pattern',
        constants: { EV_RANGE: 8.0, WORKGROUP_SIZE },
      },
    });

    const presentModule = device.createShaderModule({ code: presentWgsl, label: 'present' });
    this.presentPipeline = device.createRenderPipeline({
      label: 'present',
      layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: presentModule,
        entryPoint: 'fs_present',
        targets: [{ format: gpu.canvasFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.presentUniforms = device.createBuffer({
      size: 16, // exposure f32, padded to uniform alignment
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'present-uniforms',
    });
    this.setExposure(1.0);

    this.hdrTexture = this.createHdrTexture(width, height);
    this.hdrView = this.hdrTexture.createView();
    this.rebuildBindGroups();
  }

  private createHdrTexture(width: number, height: number): GPUTexture {
    return this.gpu.device.createTexture({
      label: 'hdr-target',
      size: { width: Math.max(1, width), height: Math.max(1, height) },
      format: HDR_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  private rebuildBindGroups(): void {
    const { device } = this.gpu;
    this.patternBindGroup = device.createBindGroup({
      layout: this.patternPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.hdrView }],
    });
    this.presentBindGroup = device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.hdrView },
        { binding: 1, resource: { buffer: this.presentUniforms } },
      ],
    });
  }

  setExposure(exposure: number): void {
    this.gpu.device.queue.writeBuffer(this.presentUniforms, 0, new Float32Array([exposure, 0, 0, 0]));
  }

  resize(width: number, height: number): void {
    if (this.hdrTexture.width === width && this.hdrTexture.height === height) return;
    this.hdrTexture.destroy();
    this.hdrTexture = this.createHdrTexture(width, height);
    this.hdrView = this.hdrTexture.createView();
    this.rebuildBindGroups();
    this.hdrStale = true;
  }

  renderFrame(): void {
    const { device, context } = this.gpu;
    const encoder = device.createCommandEncoder({ label: 'frame' });

    // The static pattern only needs recompute after (re)size — mirrors the
    // accumulate-vs-restart contract of ARCHITECTURE.md §6.
    if (this.hdrStale) {
      const pass = encoder.beginComputePass({
        label: 'test-pattern',
        timestampWrites: this.profiler.timestampWrites('pattern') as GPUComputePassTimestampWrites,
      });
      pass.setPipeline(this.patternPipeline);
      pass.setBindGroup(0, this.patternBindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(this.hdrTexture.width / WORKGROUP_SIZE),
        Math.ceil(this.hdrTexture.height / WORKGROUP_SIZE),
      );
      pass.end();
      this.hdrStale = false;
    }

    const present = encoder.beginRenderPass({
      label: 'present',
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: 'store',
        },
      ],
      timestampWrites: this.profiler.timestampWrites('present') as GPURenderPassTimestampWrites,
    });
    present.setPipeline(this.presentPipeline);
    present.setBindGroup(0, this.presentBindGroup);
    present.draw(3);
    present.end();

    this.profiler.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    this.profiler.endFrame();
  }

  destroy(): void {
    this.hdrTexture.destroy();
    this.presentUniforms.destroy();
  }
}
