/**
 * Frame orchestrator (ARCHITECTURE.md §4): owns the HDR target and Pass D
 * (optional bloom → present), delegates Passes A-C to TracePass.
 */
import type { ObserverType } from '../core/parameters';
import type { GpuContext } from '../gpu/device';
import { GpuProfiler } from '../gpu/profiler';
import presentWgsl from '../shaders/post/present.wgsl';
import { BloomPass } from './bloomPass';
import { TracePass, type CameraPose } from './tracePass';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
/** Export/readback target format (PNG bytes come out in RGBA order). */
const EXPORT_FORMAT: GPUTextureFormat = 'rgba8unorm';

/** IEEE half → float (readback decode; subnormals and infinities included). */
function decodeF16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

export class Renderer {
  private readonly gpu: GpuContext;
  readonly profiler: GpuProfiler;
  readonly tracePass: TracePass;

  private readonly presentPipeline: GPURenderPipeline;
  /** Same present transform targeting rgba8unorm (screenshot/tiled export). */
  private readonly exportPresentPipeline: GPURenderPipeline;
  private readonly presentUniforms: GPUBuffer;
  private readonly presentUniformData = new Float32Array([1, 0, 1, 0]);
  private readonly bloomPass: BloomPass;
  private bloomView: GPUTextureView;
  private hdrTexture: GPUTexture;
  private hdrView: GPUTextureView;
  private presentBindGroup!: GPUBindGroup;
  private exportPresentBindGroup!: GPUBindGroup;
  private bloomStrength = 0;

  constructor(gpu: GpuContext, width: number, height: number, marchWorkgroupSize?: number) {
    this.gpu = gpu;
    this.profiler = new GpuProfiler(gpu.device, gpu.hasTimestampQuery);
    const { device } = gpu;

    const presentModule = device.createShaderModule({ code: presentWgsl, label: 'present' });
    const makePresentPipeline = (format: GPUTextureFormat, label: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label,
        layout: 'auto',
        vertex: { module: presentModule, entryPoint: 'vs_fullscreen' },
        fragment: { module: presentModule, entryPoint: 'fs_present', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
    this.presentPipeline = makePresentPipeline(gpu.canvasFormat, 'present');
    this.exportPresentPipeline = makePresentPipeline(EXPORT_FORMAT, 'present-export');
    this.presentUniforms = device.createBuffer({
      size: 16, // exposure, bloomStrength, bloomNorm, pad
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'present-uniforms',
    });

    this.bloomPass = new BloomPass(gpu);
    this.hdrTexture = this.createHdrTexture(width, height);
    this.hdrView = this.hdrTexture.createView();
    this.bloomView = this.bloomPass.configure(this.hdrView, width, height);
    this.setExposure(1.0);
    this.rebuildPresentBindGroups();
    this.tracePass = new TracePass(gpu, this.profiler, this.hdrView, width, height, marchWorkgroupSize);
  }

  private createHdrTexture(width: number, height: number): GPUTexture {
    return this.gpu.device.createTexture({
      label: 'hdr-target',
      size: { width: Math.max(1, width), height: Math.max(1, height) },
      format: HDR_FORMAT,
      usage:
        GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
  }

  /**
   * Pre-tonemap HDR readback (validation and tiled export — V-I3
   * beaming-asymmetry check against the float64 oracle; VALIDATION.md §3).
   * Returns RGBA float rows.
   */
  async readHdr(): Promise<{ data: Float32Array; width: number; height: number }> {
    const { device } = this.gpu;
    const width = this.hdrTexture.width;
    const height = this.hdrTexture.height;
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
    const staging = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'hdr-readback',
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture: this.hdrTexture },
      { buffer: staging, bytesPerRow, rowsPerImage: height },
      { width, height },
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    const data = new Float32Array(width * height * 4);
    const halfRow = bytesPerRow / 2;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width * 4; x++) {
        data[y * width * 4 + x] = decodeF16(raw[y * halfRow + x]!);
      }
    }
    return { data, width, height };
  }

  private rebuildPresentBindGroups(): void {
    const make = (pipeline: GPURenderPipeline): GPUBindGroup =>
      this.gpu.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.hdrView },
          { binding: 1, resource: { buffer: this.presentUniforms } },
          { binding: 2, resource: this.bloomView },
          { binding: 3, resource: this.bloomSampler() },
        ],
      });
    this.presentBindGroup = make(this.presentPipeline);
    this.exportPresentBindGroup = make(this.exportPresentPipeline);
  }

  private presentSampler: GPUSampler | null = null;
  private bloomSampler(): GPUSampler {
    this.presentSampler ??= this.gpu.device.createSampler({
      label: 'present-bloom-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    return this.presentSampler;
  }

  private writePresentUniforms(): void {
    this.presentUniformData[1] = this.bloomStrength;
    this.presentUniformData[2] = 1 / Math.max(1, this.bloomPass.levelCount);
    this.gpu.device.queue.writeBuffer(this.presentUniforms, 0, this.presentUniformData);
  }

  setExposure(exposure: number): void {
    this.presentUniformData[0] = exposure;
    this.writePresentUniforms();
  }

  /** M7 optional bloom (camera model; display-only — no accumulation restart). */
  setBloomStrength(strength: number): void {
    this.bloomStrength = strength;
    this.writePresentUniforms();
  }

  setCamera(pose: CameraPose, spin = 0, observer: ObserverType = 'zamo'): void {
    this.tracePass.setCamera(pose, spin, observer);
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
    this.hdrView = this.hdrTexture.createView();
    this.bloomView = this.bloomPass.configure(this.hdrView, width, height);
    this.writePresentUniforms();
    this.rebuildPresentBindGroups();
    this.tracePass.resize(width, height, this.hdrView);
  }

  private encodePresent(
    encoder: GPUCommandEncoder,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    view: GPUTextureView,
    withTimestamps: boolean,
  ): void {
    const present = encoder.beginRenderPass({
      label: 'present',
      colorAttachments: [
        { view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' },
      ],
      timestampWrites: (withTimestamps
        ? this.profiler.timestampWrites('present')
        : undefined) as GPURenderPassTimestampWrites,
    });
    present.setPipeline(pipeline);
    present.setBindGroup(0, bindGroup);
    present.draw(3);
    present.end();
  }

  renderFrame(): void {
    const { device, context } = this.gpu;
    const encoder = device.createCommandEncoder({ label: 'frame' });

    this.tracePass.encode(encoder);
    if (this.bloomStrength > 0) {
      this.bloomPass.encode(
        encoder,
        this.profiler.timestampWrites('bloom') as GPUComputePassTimestampWrites,
      );
    }
    this.encodePresent(
      encoder,
      this.presentPipeline,
      this.presentBindGroup,
      context.getCurrentTexture().createView(),
      true,
    );

    this.profiler.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    this.tracePass.afterSubmit();
    this.profiler.endFrame();
  }

  /** Trace-only frame (M7 tiled export): no canvas present, no profiler churn. */
  renderHeadless(): void {
    const { device } = this.gpu;
    const encoder = device.createCommandEncoder({ label: 'headless-frame' });
    this.tracePass.encode(encoder);
    device.queue.submit([encoder.finish()]);
    this.tracePass.afterSubmit();
  }

  /**
   * Run the present transform (exposure → AgX → sRGB; bloom per current
   * strength) into an rgba8 texture and read the bytes back — the PNG
   * screenshot and tiled-export path. Bloom note: the blur pyramid is only
   * refreshed by interactive frames, so exports render with the pyramid state
   * of the last presented frame; the tiled exporter forces strength 0 for its
   * tiles instead (a documented limitation — bloom kernels cross tile edges).
   */
  async presentToBytes(): Promise<{ bytes: Uint8Array; width: number; height: number }> {
    const { device } = this.gpu;
    const width = this.hdrTexture.width;
    const height = this.hdrTexture.height;
    const target = device.createTexture({
      label: 'export-present',
      size: { width, height },
      format: EXPORT_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const staging = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'export-readback',
    });
    const encoder = device.createCommandEncoder({ label: 'export-present' });
    if (this.bloomStrength > 0) {
      this.bloomPass.encode(encoder);
    }
    this.encodePresent(encoder, this.exportPresentPipeline, this.exportPresentBindGroup, target.createView(), false);
    encoder.copyTextureToBuffer(
      { texture: target },
      { buffer: staging, bytesPerRow, rowsPerImage: height },
      { width, height },
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    target.destroy();

    const bytes = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      bytes.set(raw.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
    }
    return { bytes, width, height };
  }

  destroy(): void {
    this.tracePass.destroy();
    this.bloomPass.destroy();
    this.hdrTexture.destroy();
    this.presentUniforms.destroy();
  }
}
