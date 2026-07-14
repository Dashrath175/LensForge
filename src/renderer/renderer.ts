/**
 * Frame orchestrator (ARCHITECTURE.md §4): owns the HDR target and Pass D
 * (present), delegates Passes A-C to TracePass.
 */
import type { GpuContext } from '../gpu/device';
import { GpuProfiler } from '../gpu/profiler';
import presentWgsl from '../shaders/post/present.wgsl';
import { TracePass, type CameraPose } from './tracePass';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';

export class Renderer {
  private readonly gpu: GpuContext;
  readonly profiler: GpuProfiler;
  readonly tracePass: TracePass;

  private readonly presentPipeline: GPURenderPipeline;
  private readonly presentUniforms: GPUBuffer;
  private hdrTexture: GPUTexture;
  private presentBindGroup!: GPUBindGroup;

  constructor(gpu: GpuContext, width: number, height: number) {
    this.gpu = gpu;
    this.profiler = new GpuProfiler(gpu.device, gpu.hasTimestampQuery);
    const { device } = gpu;

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
    const hdrView = this.hdrTexture.createView();
    this.rebuildPresentBindGroup(hdrView);
    this.tracePass = new TracePass(gpu, this.profiler, hdrView, width, height);
  }

  private createHdrTexture(width: number, height: number): GPUTexture {
    return this.gpu.device.createTexture({
      label: 'hdr-target',
      size: { width: Math.max(1, width), height: Math.max(1, height) },
      format: HDR_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  private rebuildPresentBindGroup(hdrView: GPUTextureView): void {
    this.presentBindGroup = this.gpu.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: hdrView },
        { binding: 1, resource: { buffer: this.presentUniforms } },
      ],
    });
  }

  setExposure(exposure: number): void {
    this.gpu.device.queue.writeBuffer(this.presentUniforms, 0, new Float32Array([exposure, 0, 0, 0]));
  }

  setCamera(pose: CameraPose, spin = 0): void {
    this.tracePass.setCamera(pose, spin);
  }

  dimensions(): { width: number; height: number } {
    return { width: this.hdrTexture.width, height: this.hdrTexture.height };
  }

  onGpuIdle(): Promise<undefined> {
    return this.gpu.device.queue.onSubmittedWorkDone();
  }

  resize(width: number, height: number): void {
    if (this.hdrTexture.width === width && this.hdrTexture.height === height) return;
    this.hdrTexture.destroy();
    this.hdrTexture = this.createHdrTexture(width, height);
    const hdrView = this.hdrTexture.createView();
    this.rebuildPresentBindGroup(hdrView);
    this.tracePass.resize(width, height, hdrView);
  }

  renderFrame(): void {
    const { device, context } = this.gpu;
    const encoder = device.createCommandEncoder({ label: 'frame' });

    this.tracePass.encode(encoder);

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
    this.tracePass.afterSubmit();
    this.profiler.endFrame();
  }

  destroy(): void {
    this.tracePass.destroy();
    this.hdrTexture.destroy();
    this.presentUniforms.destroy();
  }
}
