/**
 * M7 optional bloom (documented camera model — see shaders/post/bloom.wgsl):
 * dual-filter downsample pyramid + tent upsample with skip additions over the
 * HDR target. The half-resolution result accumulates `levels` unit-gain blur
 * scales, so the present pass normalizes by 1/levels before the
 * energy-conserving mix. Encoded only when bloomStrength > 0 (off by default:
 * zero cost, and the zero-initialized result texture composites to nothing).
 */
import type { GpuContext } from '../gpu/device';
import bloomWgsl from '../shaders/post/bloom.wgsl';

const MAX_LEVELS = 6;
const WORKGROUP_2D = 8;

interface Level {
  down: GPUTexture;
  downView: GPUTextureView;
  up: GPUTexture | null;
  upView: GPUTextureView | null;
  width: number;
  height: number;
}

export class BloomPass {
  private readonly gpu: GpuContext;
  private readonly downPipeline: GPUComputePipeline;
  private readonly upPipeline: GPUComputePipeline;
  private readonly sampler: GPUSampler;
  private levels: Level[] = [];
  private downBindGroups: GPUBindGroup[] = [];
  private upBindGroups: GPUBindGroup[] = [];

  constructor(gpu: GpuContext) {
    this.gpu = gpu;
    const { device } = gpu;
    const module = device.createShaderModule({ code: bloomWgsl, label: 'bloom' });
    this.downPipeline = device.createComputePipeline({
      label: 'bloom-down',
      layout: 'auto',
      compute: { module, entryPoint: 'cs_down' },
    });
    this.upPipeline = device.createComputePipeline({
      label: 'bloom-up',
      layout: 'auto',
      compute: { module, entryPoint: 'cs_up' },
    });
    this.sampler = device.createSampler({
      label: 'bloom-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  /** (Re)build the pyramid for an HDR target; returns the composite view. */
  configure(hdrView: GPUTextureView, width: number, height: number): GPUTextureView {
    const { device } = this.gpu;
    for (const level of this.levels) {
      level.down.destroy();
      level.up?.destroy();
    }
    const levelCount = Math.max(
      1,
      Math.min(MAX_LEVELS, Math.floor(Math.log2(Math.min(width, height))) - 4),
    );
    this.levels = [];
    for (let i = 0; i < levelCount; i++) {
      const w = Math.max(1, width >> (i + 1));
      const h = Math.max(1, height >> (i + 1));
      const make = (label: string): GPUTexture =>
        device.createTexture({
          label,
          size: { width: w, height: h },
          format: 'rgba16float',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
      const down = make(`bloom-down-${i}`);
      // The deepest level needs no up texture (its down IS the up input).
      const up = i < levelCount - 1 ? make(`bloom-up-${i}`) : null;
      this.levels.push({
        down,
        downView: down.createView(),
        up,
        upView: up ? up.createView() : null,
        width: w,
        height: h,
      });
    }

    this.downBindGroups = this.levels.map((level, i) =>
      device.createBindGroup({
        layout: this.downPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: i === 0 ? hdrView : this.levels[i - 1]!.downView },
          { binding: 2, resource: level.downView },
        ],
      }),
    );
    // Up pass i composites tent(next-coarser result) + down[i], for i = L−2..0.
    this.upBindGroups = [];
    for (let i = 0; i < levelCount - 1; i++) {
      const coarser = i === levelCount - 2 ? this.levels[i + 1]!.downView : this.levels[i + 1]!.upView!;
      this.upBindGroups.push(
        device.createBindGroup({
          layout: this.upPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.sampler },
            { binding: 1, resource: coarser },
            { binding: 2, resource: this.levels[i]!.upView ?? this.levels[i]!.downView },
            { binding: 3, resource: this.levels[i]!.downView },
          ],
        }),
      );
    }
    return this.resultView();
  }

  /** Half-resolution composite of all blur scales (normalize by levelCount). */
  resultView(): GPUTextureView {
    const first = this.levels[0]!;
    return first.upView ?? first.downView;
  }

  get levelCount(): number {
    return this.levels.length;
  }

  encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void {
    const pass = encoder.beginComputePass({
      label: 'bloom',
      ...(timestampWrites ? { timestampWrites } : {}),
    });
    pass.setPipeline(this.downPipeline);
    this.levels.forEach((level, i) => {
      pass.setBindGroup(0, this.downBindGroups[i]!);
      pass.dispatchWorkgroups(
        Math.ceil(level.width / WORKGROUP_2D),
        Math.ceil(level.height / WORKGROUP_2D),
      );
    });
    pass.setPipeline(this.upPipeline);
    for (let i = this.levels.length - 2; i >= 0; i--) {
      const level = this.levels[i]!;
      pass.setBindGroup(0, this.upBindGroups[i]!);
      pass.dispatchWorkgroups(
        Math.ceil(level.width / WORKGROUP_2D),
        Math.ceil(level.height / WORKGROUP_2D),
      );
    }
    pass.end();
  }

  destroy(): void {
    for (const level of this.levels) {
      level.down.destroy();
      level.up?.destroy();
    }
    this.levels = [];
  }
}
