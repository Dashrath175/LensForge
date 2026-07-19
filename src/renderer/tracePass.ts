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
import { diskEffectiveTemperatureK } from '../core/units';
import type { GpuContext } from '../gpu/device';
import type { GpuProfiler } from '../gpu/profiler';
import { GEODESIC_DEFAULTS } from '../physics/integrator';
import type { ObserverType } from '../core/parameters';
import { createMetric, evalMetric, horizonRadius } from '../physics/metric';
import { iscoRadius } from '../physics/orbits';
import { observerFourVelocity, orthonormalTetrad, type Tetrad } from '../physics/tetrad';
import initWgsl from '../shaders/trace/init.wgsl';
import marchWgsl from '../shaders/trace/march.wgsl';
import shadeWgsl from '../shaders/trace/shade.wgsl';
import {
  samplePlanckXyz,
  sliceNtAtSpin,
  type DiskLuts,
} from './diskLuts';

// Mirrors src/shaders/common/ray-state.wgsl — change both together.
export const STATUS_MARCHING = 0;
export const STATUS_CAPTURED = 1;
export const STATUS_ESCAPED = 2;
export const STATUS_MAX_STEPS = 3;
export const STATUS_HIT_DISK = 4;
const STATE_PLANES = 11;
const RESULT_PLANES = 3;

const UNIFORM_BYTES = 224;

/**
 * Temporal accumulation on camera rest (M6, IMPLEMENTATION.md): after the
 * base trace converges (sample 0, exact pixel centers — the image every
 * V-O/V-I comparison uses), refinement rounds re-trace the frame with
 * Halton-jittered sub-pixel offsets and fold each converged round into a
 * float32 running mean. Every jittered ray is an exact geodesic; the mean is
 * the physically correct pixel-footprint integral (box filter) — still-frame
 * edge quality (shadow boundary, photon ring, graticule) refines without
 * touching physics. Any camera/parameter change restarts at the base sample.
 */
const REFINE_MAX_ROUNDS = 31; // 32 samples total including the base
/** Fallback march workgroup size; production value comes from the startup
 * micro-benchmark (src/renderer/marchBenchmark.ts, M6). */
const WORKGROUP_1D_DEFAULT = 64;
const WORKGROUP_2D = 8;

/** Display peak for the temperature-scaled color mode (PHYSICS.md §8): the
 * disk's maximum T_obs at g = 1 is mapped to this visible-locus temperature. */
const TEMP_SCALED_PEAK_K = 6500;
/** HDR working-range luminance assigned to the disk's g = 1 temperature peak.
 * A global linear pre-exposure (camera model): keeps the beaming dynamic range
 * (~10³-10⁴, D10) inside rgba16f with headroom. */
const HDR_PEAK_Y = 4.0;

/**
 * The single quality knob (D11): integrator tolerance per tier, tuned against
 * the V-O1 bounds (1/3/10 arcmin). Escape-direction error scales as
 * ~steps × rtol × |y|, so High tier sits near the f32 truncation floor.
 */
export const QUALITY_TIERS = {
  high: { rtol: 1e-6, atol: 1e-8 },
  medium: { rtol: 1e-5, atol: 1e-7 },
  // M6 per-tier V-O1 rerun: rtol 1e-4 measured 17-32 arcmin — outside the
  // documented 10-arcmin Low bound. 3e-5 measures ≤ ~6 arcmin (validated).
  low: { rtol: 3e-5, atol: 3e-7 },
} as const;
export type QualityTier = keyof typeof QUALITY_TIERS;

/** DP45 attempts per ray per dispatch; bounds dispatch duration (V-P4). */
const STEPS_PER_DISPATCH = 64;

/**
 * Frame-budget governor (M6, "chunk budget K from measured timings"): the
 * scheduler multiplies the scaled chunk baselines by workScale, a feedback
 * control on the measured total trace GPU time per frame (timestamp-query
 * EMAs). Fast GPUs converge faster and show more full-resolution detail
 * during interaction; slow GPUs shed the scalable budgets first — the
 * coarsest preview level's completeness ladder is never scaled (the
 * presented-image-complete guarantee, ARCHITECTURE.md §5). Without
 * timestamp-query the baselines apply unscaled.
 */
const GOVERNOR_TARGET_TRACE_MS = 9;
const GOVERNOR_SCALE_MIN = 0.5;
const GOVERNOR_SCALE_MAX = 8;
/** Per-frame multiplicative step bounds — settles in ~a second, no ringing. */
const GOVERNOR_STEP_DOWN = 0.85;
const GOVERNOR_STEP_UP = 1.08;
/** Full-resolution march chunks per frame at workScale = 1 (scaled, clamped).
 * These run on interaction frames too: rays that terminate within the
 * in-frame ladder present at full resolution immediately — sky, graticule
 * and disk-body detail stays sharp during motion. */
const MAIN_CHUNKS_BASE = 2;
const MAIN_CHUNKS_MAX = 16;
/**
 * Interaction preview ladder (ARCHITECTURE.md §5 — the presented image must
 * be complete for the current pose every frame). Coarse instances of the
 * same A-B-C trace run at descending scales, coarsest first; each level's
 * Pass C presents the next-coarser level's radiance for its own
 * still-MARCHING pixels (the coarsest binds a 1×1 black dummy), and the
 * full-resolution shade presents the finest level. Same shaders, same
 * physics, same pose at every level — never stale frames, never black holes.
 *
 * Budgets (M5 baseline; the M6 governor adapts them): the coarsest level's
 * chunk ladder terminates all but the near-critical winding set within one
 * frame — the completeness guarantee. The half-resolution level gets a
 * smaller budget that covers the fast bulk of rays, so during motion most of
 * the frame presents ½-resolution detail and only the slow tail falls back
 * to ¼ — progressive rendering stays visually near the converged image.
 */
interface PreviewLevelConfig {
  scale: number;
  /** Chunk budget at workScale = 1; scaled levels track the governor. */
  dispatchesPerFrame: number;
  /** Whether the governor may scale this level's budget. The coarsest level
   * is never scaled — its in-frame convergence is the completeness
   * guarantee behind every finer level. */
  governed: boolean;
}
const PREVIEW_LEVELS: readonly PreviewLevelConfig[] = [
  { scale: 4, dispatchesPerFrame: 32, governed: false },
  { scale: 2, dispatchesPerFrame: 12, governed: true },
];
/** Governed chunk bounds for preview levels. */
const PREVIEW_CHUNKS_MIN = 4;
const PREVIEW_CHUNKS_MAX = 48;
/**
 * Sky radiance scale (M7 camera model, documented like HDR_PEAK_Y): maps the
 * shipped cubemap's normalized units (sphere-mean channel radiance = 1,
 * tools/generateSkyAssets.ts) into the HDR working range where the disk's
 * g = 1 peak sits at HDR_PEAK_Y. 0.02 puts the Milky Way band (p99 ≈ 7 R+G+B)
 * near the placeholder sky's band level and first-magnitude star splats at
 * roughly the disk's tone-mapped shoulder. Relative sky photometry —
 * background vs. star vs. star — is fixed by the shared panorama units and is
 * NOT affected by this single dial.
 */
const SKY_RADIANCE_SCALE = 0.02;
/** Star-splat PSF sigma in screen pixels (shaders/sky/sky.wgsl): wide enough
 * that pixel-center sampling of the Gaussian is alias-free (no twinkling on
 * the base sample); refinement rounds integrate it exactly. */
const STAR_PSF_SIGMA_PX = 0.8;
/** Star-grid dimensions — mirrored in sky.wgsl and tools/generateSkyAssets.ts. */
const SKY_GRID_CELLS = 32 * 64;

export interface CameraPose {
  r: number;
  theta: number;
  phi: number;
  fovY: number;
  /** Free-look offsets (M5): spatial rotation of the local view triad.
   * Omitted/zero = view axis points at the hole (orbit convention). */
  yaw?: number;
  pitch?: number;
}

/** Novikov-Thorne disk parameters (M4). Physical scales enter only here (D5). */
export interface DiskParams {
  enabled: boolean;
  /** Outer edge in GM/c² (inner edge is always the ISCO — zero-torque truncation). */
  rOuter: number;
  massKg: number;
  mdotKgPerS: number;
  /** PHYSICS.md §8: 'physical' = true T_eff; 'scaled' = global T rescale into the visible locus. */
  colorMode: 'physical' | 'scaled';
  /** M7 statistical inhomogeneity amplitude, 0 = off (PHYSICS.md §9.6):
   * relative flux contrast of a mean-1 brightness field advected at the
   * exact Ω(r). Labeled "statistical model" in the UI. */
  inhomogeneity: number;
}

export const DEFAULT_DISK: DiskParams = {
  enabled: true,
  rOuter: 20,
  massKg: 1.989e37, // 1e7 solar masses
  mdotKgPerS: 1.4e21, // ~0.1 Eddington at 10% efficiency for that mass
  colorMode: 'scaled',
  inhomogeneity: 0,
};

/** Tile spec for the tiled high-res export (M7): this trace renders the
 * pixel rectangle starting at (x, y) of a fullWidth × fullHeight image. */
export interface TileSpec {
  x: number;
  y: number;
  fullWidth: number;
  fullHeight: number;
}

/** Sky assets uploaded by setSky (loaded in src/renderer/skyAssets.ts). */
export interface SkyAssetData {
  faceSize: number;
  /** 6 faces of rgba16f texels, tightly packed, +X,−X,+Y,−Y,+Z,−Z. */
  cubemap: Uint8Array;
  aggFaceSize: number;
  aggCubemap: Uint8Array;
  /** f32 star records [dir xyz, flux rgb, 0, 0] sorted by grid cell. */
  starRecords: Float32Array<ArrayBuffer>;
  /** CSR cell offsets, length SKY_GRID_CELLS + 1. */
  cellStart: Uint32Array<ArrayBuffer>;
}

export interface BatchResult {
  statuses: Uint32Array;
  /** Final geodesic states, 8 components per ray (f32 lifted to f64); at a
   * disk hit this is the Hermite-refined crossing state. */
  states: Float64Array;
  /** Per-ray result payload, 3 values: ESCAPED → asymptotic direction xyz;
   * HIT_DISK → (r_hit, φ_hit, g). */
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

/**
 * One resident horizontal band of a trace grid (adapter-limit compliance):
 * no storage buffer may exceed the granted maxStorageBufferBindingSize /
 * maxBufferSize, so a grid whose per-ray buffers would (the state plane is
 * 44 B/pixel — a fullscreen canvas overruns even 1 GiB grants at 8K) is
 * split into slabs. Each slab is an independent tile of the same image
 * through the M7 tile-uniform path — the mapping measured bit-identical to
 * a monolithic render by the tiled-export consistency check. Interior slab
 * borders carry a 1-row apron (the export-tile apron) so shade-pass
 * star-splat Jacobians keep central differences; apron rows are traced but
 * never composed into the target.
 */
interface Slab {
  /** First interior row inside the grid image. */
  y0: number;
  /** Interior rows (composed into the target). */
  rows: number;
  /** Apron rows above the interior (0 or 1; bottom apron is implied by gridH). */
  apronTop: number;
  /** Slab grid height including aprons; width always equals the grid width. */
  gridH: number;
  buffers: RayBuffers;
  /** Per-slab accumulation state (16 B/ray — same layout as before, banded). */
  accum: GPUBuffer;
  /** Per-slab uniforms: the canonical block with this slab's tile words. */
  uniformBuffer: GPUBuffer;
  /** Own shade target when the grid is split (composed by copy); null for a
   * single-slab grid, which shades straight into the grid target — the
   * pre-split code path, byte-for-byte. */
  texture: GPUTexture | null;
  view: GPUTextureView | null;
  initBindGroup: GPUBindGroup | null;
  marchBindGroup: GPUBindGroup | null;
  shadeBindGroup: GPUBindGroup | null;
}

/** A pixel grid traced as one or more slabs (screen and preview levels). */
interface SlabGrid {
  width: number;
  height: number;
  /** Σ slab rayCounts — the marching-counter start value (apron rays are
   * duplicated across adjacent slabs and counted per slab; the counter is a
   * countdown to zero, so duplicates only shift the intermediate readings). */
  rayCount: number;
  slabs: Slab[];
}

/** One rung of the interaction preview ladder (see PREVIEW_LEVELS). */
interface PreviewLevel {
  readonly config: PreviewLevelConfig;
  readonly counter: GPUBuffer;
  readonly counterStaging: GPUBuffer;
  width: number;
  height: number;
  grid: SlabGrid | null;
  texture: GPUTexture | null;
  view: GPUTextureView | null;
  knownMarching: number;
  mapPending: boolean;
  copyEncoded: boolean;
}

/**
 * Free-look (M5): rotate the spatial triad {e1, e2, e3} by yaw about e2
 * (positive = look toward +e3, screen right) then pitch about the yawed e3
 * (positive = look toward -e2, screen up; init.wgsl maps view = -e1,
 * up = -e2, right = +e3). An SO(3) mix of orthonormal spatial legs is again
 * an orthonormal tetrad for the same observer — aberration is untouched,
 * orientation is exact.
 */
function rotateTetradLook(t: Tetrad, yaw: number, pitch: number): Tetrad {
  if (yaw === 0 && pitch === 0) return t;
  const [e0, e1, e2, e3] = t;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const mix = (a: number[], ca: number, b: number[], cb: number): [number, number, number, number] =>
    [0, 1, 2, 3].map((i) => ca * a[i]! + cb * b[i]!) as [number, number, number, number];
  const e1a = mix(e1, cy, e3, -sy);
  const e3a = mix(e1, sy, e3, cy);
  const e1b = mix(e1a, cp, e2, sp);
  const e2b = mix(e1a, -sp, e2, cp);
  return [e0, e1b, e2b, e3a] as Tetrad;
}

/** FNV-1a over the packed uniform bytes (dev-only accumulation invariant). */
function hashUniforms(data: ArrayBuffer): number {
  const bytes = new Uint8Array(data);
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export class TracePass {
  private readonly gpu: GpuContext;
  private readonly profiler: GpuProfiler;

  private readonly initPipeline: GPUComputePipeline;
  private readonly marchPipeline: GPUComputePipeline;
  private readonly shadePipeline: GPUComputePipeline;

  /** Canonical full-image uniform block (CPU side): per-slab GPU uniform
   * buffers are derived from it by overriding only the tile/grid words. */
  private readonly uniformData = new ArrayBuffer(UNIFORM_BYTES);
  private readonly counter: GPUBuffer;
  private readonly counterStaging: GPUBuffer;
  private counterMapPending = false;
  /** Set when encode() copied the counter to staging this frame; a readback
   * may only map staging after a fresh copy, or it would resurrect a count
   * from an older generation (falsely zeroing knownMarching stalls the march). */
  private counterCopyEncoded = false;

  private screenGrid: SlabGrid | null = null;
  /** Dev/validation override of the per-buffer byte budget (createGrid): lets
   * the slab path be exercised and bit-compared at ordinary resolutions. */
  private maxStorageBytesForTest: number | null = null;

  // Interaction preview ladder (see PREVIEW_LEVELS), coarsest first: per
  // level a coarse ray state, its own uniform buffer (same camera words,
  // level pixel-grid words), its own marching counter, and an rgba16f target
  // sampled by the next-finer level (finest: by the full-resolution shade).
  private readonly levels: PreviewLevel[] = [];
  private readonly fallbackSampler: GPUSampler;
  /** 1×1 black texture bound as the coarsest level's fallback. */
  private readonly fallbackDummy: GPUTexture;

  // Float32 accumulation state (M6) lives per slab (Slab.accum): the running
  // mean of jittered samples. Preview slabs hold their own — they always
  // write with weight 1 while the base trace is unconverged, so their reads
  // never contribute (mix with weight 1 ignores the accumulator).
  private refineRound = 0;
  private refineActive = false;
  private refineMarching = 0;
  private refinementEnabled = true;
  /** Bumped on camera changes AND round starts; counter readbacks initiated
   * under an older epoch are discarded (same hazard class as generation). */
  private epoch = 0;

  private width = 0;
  private height = 0;
  private hdrTexture!: GPUTexture;
  private hdrView!: GPUTextureView;
  private camera: CameraPose | null = null;
  private spin = 0;
  private observer: ObserverType = 'zamo';
  /** Simulation time driving the inhomogeneity advection (M7); a uniform
   * word, so changes restart the accumulation like any physics change. */
  private diskTime = 0;
  /** Interaction preview ladder toggle: disabled during tiled export, where
   * only the converged full-resolution result is ever read. */
  private previewsEnabled = true;

  // M7 sky resources (dummies until setSky uploads the shipped assets:
  // black cubemaps + an empty star grid render a starless black sky).
  private skyTexture: GPUTexture;
  private skyView: GPUTextureView;
  private skyAggTexture: GPUTexture;
  private skyAggView: GPUTextureView;
  private readonly skySampler: GPUSampler;
  private starBuffer: GPUBuffer;
  private cellStartBuffer: GPUBuffer;
  private tier: QualityTier = 'high';
  private disk: DiskParams = { ...DEFAULT_DISK };
  private luts: DiskLuts | null = null;
  /** Current-spin T̂ slice (CPU copy shared with the validation harness). */
  private ntSlice: Float32Array | null = null;
  private readonly ntSliceBuffer: GPUBuffer;
  private readonly planckBuffer: GPUBuffer;
  private needsInit = false;
  private knownMarching = 0;
  /** Bumped on every camera change; stale counter readbacks (initiated under
   * an older generation) are discarded, or they could zero knownMarching
   * right after a re-init and stall the march with an unconverged image. */
  private generation = 0;

  /** Benchmark-selected march workgroup size (see marchBenchmark.ts). */
  private readonly marchWorkgroup: number;

  constructor(
    gpu: GpuContext,
    profiler: GpuProfiler,
    hdrTarget: { texture: GPUTexture; view: GPUTextureView },
    width: number,
    height: number,
    marchWorkgroupSize: number = WORKGROUP_1D_DEFAULT,
  ) {
    this.gpu = gpu;
    this.profiler = profiler;
    this.marchWorkgroup = marchWorkgroupSize;
    const { device } = gpu;

    const make = (code: string, entryPoint: string, label: string): GPUComputePipeline =>
      device.createComputePipeline({
        label,
        layout: 'auto',
        compute: { module: device.createShaderModule({ code, label }), entryPoint },
      });
    this.initPipeline = make(initWgsl, 'cs_init', 'trace-init');
    this.marchPipeline = device.createComputePipeline({
      label: 'trace-march',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: marchWgsl, label: 'trace-march' }),
        entryPoint: 'cs_march',
        constants: { WORKGROUP_SIZE: this.marchWorkgroup },
      },
    });
    this.shadePipeline = make(shadeWgsl, 'cs_shade', 'trace-shade');

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
    for (const config of PREVIEW_LEVELS) {
      this.levels.push({
        config,
        counter: device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          label: `preview${config.scale}-marching-counter`,
        }),
        counterStaging: device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          label: `preview${config.scale}-marching-counter-staging`,
        }),
        width: 0,
        height: 0,
        grid: null,
        texture: null,
        view: null,
        knownMarching: 0,
        mapPending: false,
        copyEncoded: false,
      });
    }
    this.fallbackSampler = device.createSampler({
      label: 'preview-fallback',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.fallbackDummy = device.createTexture({
      label: 'fallback-dummy',
      size: { width: 1, height: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    // LUT buffers sized to the tools/ generator constants (asserted on load);
    // zero until setLuts, and the disk stays disabled until then.
    this.ntSliceBuffer = device.createBuffer({
      size: 512 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'nt-teff-slice',
    });
    this.planckBuffer = device.createBuffer({
      size: 512 * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'planck-xyz',
    });

    // Sky dummies (replaced by setSky): 1×1×6 black cubemaps, empty star grid.
    const makeCubeDummy = (label: string): GPUTexture =>
      device.createTexture({
        label,
        size: { width: 1, height: 1, depthOrArrayLayers: 6 },
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING,
      });
    this.skySampler = device.createSampler({
      label: 'sky-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    this.skyTexture = makeCubeDummy('sky-cubemap-dummy');
    this.skyView = this.skyTexture.createView({ dimension: 'cube' });
    this.skyAggTexture = makeCubeDummy('star-agg-dummy');
    this.skyAggView = this.skyAggTexture.createView({ dimension: 'cube' });
    this.starBuffer = device.createBuffer({
      size: 8 * 4,
      usage: GPUBufferUsage.STORAGE,
      label: 'sky-stars-dummy',
    });
    this.cellStartBuffer = device.createBuffer({
      size: (SKY_GRID_CELLS + 1) * 4,
      usage: GPUBufferUsage.STORAGE,
      label: 'sky-cellstart-dummy',
    });

    this.resize(width, height, hdrTarget);
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

  /** Per-buffer byte budget the GRANTED device limits allow (see Slab). */
  private bufferByteBudget(): number {
    const limits = this.gpu.device.limits;
    return Math.min(
      limits.maxStorageBufferBindingSize,
      limits.maxBufferSize,
      this.maxStorageBytesForTest ?? Number.MAX_SAFE_INTEGER,
    );
  }

  /**
   * Build a trace grid as the minimum number of horizontal slabs whose every
   * storage buffer fits the granted limits (see the Slab contract above).
   * A grid that fits in one slab reproduces the pre-split allocation and
   * dispatch pattern byte-for-byte.
   */
  private createGrid(width: number, height: number, label: string): SlabGrid {
    const { device } = this.gpu;
    // The state plane set (44 B/ray) is the widest per-ray buffer; accum
    // (16), result (12), and status (4) then fit automatically. The 1D march
    // dispatch bounds rays per slab too: ceil(rays / workgroup) must not
    // exceed maxComputeWorkgroupsPerDimension (65535 — a 3840×2160 canvas at
    // workgroup 64 already needs 129600 groups; an invalid pass is silently
    // skipped, which presented as a black frame and a falsely-zero marching
    // counter at fullscreen).
    const maxRays = Math.min(
      Math.floor(this.bufferByteBudget() / (STATE_PLANES * 4)),
      device.limits.maxComputeWorkgroupsPerDimension * this.marchWorkgroup,
    );
    const capacityRows = Math.floor(maxRays / Math.max(1, width));
    if (capacityRows < 3) {
      throw new Error(
        `LensForge: adapter storage limits cannot hold three ${width}-pixel rows of ray state ` +
          `(budget ${this.bufferByteBudget()} B) — the canvas is too wide for this GPU.`,
      );
    }
    const slabCount = Math.max(1, Math.ceil(height / (capacityRows - 2)));
    const interior = Math.ceil(height / slabCount);
    const slabs: Slab[] = [];
    for (let y0 = 0; y0 < height; y0 += interior) {
      const rows = Math.min(interior, height - y0);
      const apronTop = y0 > 0 ? 1 : 0;
      const apronBottom = y0 + rows < height ? 1 : 0;
      const gridH = rows + apronTop + apronBottom;
      const rayCount = width * gridH;
      const slabLabel = slabCount > 1 ? `${label}-y${y0}` : label;
      const texture =
        slabCount > 1
          ? device.createTexture({
              label: `${slabLabel}-target`,
              size: { width, height: gridH },
              format: 'rgba16float',
              usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
            })
          : null;
      slabs.push({
        y0,
        rows,
        apronTop,
        gridH,
        buffers: this.createRayBuffers(rayCount, slabLabel),
        accum: device.createBuffer({
          size: rayCount * 16,
          usage: GPUBufferUsage.STORAGE,
          label: `${slabLabel}-accum`,
        }),
        uniformBuffer: device.createBuffer({
          size: UNIFORM_BYTES,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          label: `${slabLabel}-uniforms`,
        }),
        texture,
        view: texture ? texture.createView() : null,
        initBindGroup: null,
        marchBindGroup: null,
        shadeBindGroup: null,
      });
    }
    return { width, height, rayCount: slabs.reduce((s, x) => s + x.buffers.rayCount, 0), slabs };
  }

  private destroyGrid(grid: SlabGrid): void {
    for (const slab of grid.slabs) {
      this.destroyRayBuffers(slab.buffers);
      slab.accum.destroy();
      slab.uniformBuffer.destroy();
      slab.texture?.destroy();
    }
  }

  resize(width: number, height: number, hdrTarget: { texture: GPUTexture; view: GPUTextureView }): void {
    this.width = width;
    this.height = height;
    this.hdrTexture = hdrTarget.texture;
    this.hdrView = hdrTarget.view;
    if (this.screenGrid) this.destroyGrid(this.screenGrid);
    this.screenGrid = this.createGrid(width, height, 'screen');
    for (const level of this.levels) {
      level.width = Math.max(1, Math.ceil(width / level.config.scale));
      level.height = Math.max(1, Math.ceil(height / level.config.scale));
      if (level.grid) this.destroyGrid(level.grid);
      level.grid = this.createGrid(level.width, level.height, `preview${level.config.scale}`);
      level.texture?.destroy();
      level.texture = this.gpu.device.createTexture({
        label: `preview${level.config.scale}-target`,
        size: { width: level.width, height: level.height },
        format: 'rgba16float',
        // COPY_DST: composition target when the level grid is slab-split.
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      level.view = level.texture.createView();
    }

    this.buildBindGroups();

    if (this.camera) this.setCamera(this.camera, this.spin, this.observer);
  }

  /**
   * Dev/validation hook: cap the per-buffer byte budget so the slab path can
   * be exercised and bit-compared against the monolithic path at ordinary
   * resolutions (the M7 tile-consistency methodology). null restores the
   * granted device limits. Rebuilds the grids; restarts the accumulation.
   */
  setMaxStorageBytesForTest(bytes: number | null): void {
    this.maxStorageBytesForTest = bytes;
    this.resize(this.width, this.height, { texture: this.hdrTexture, view: this.hdrView });
  }

  /** Slab counts per grid (validation introspection). */
  getSlabLayout(): { screen: number; levels: number[] } {
    return {
      screen: this.screenGrid?.slabs.length ?? 0,
      levels: this.levels.map((l) => l.grid?.slabs.length ?? 0),
    };
  }

  /** (Re)create every bind group from the current buffers/textures. Called on
   * resize and when the sky assets arrive (setSky swaps the sky resources). */
  private buildBindGroups(): void {
    const { device } = this.gpu;
    const makeInit = (u: GPUBindGroupEntry, b: RayBuffers): GPUBindGroup =>
      device.createBindGroup({
        layout: this.initPipeline.getBindGroupLayout(0),
        entries: [
          u,
          { binding: 1, resource: { buffer: b.state } },
          { binding: 2, resource: { buffer: b.status } },
        ],
      });
    const makeMarch = (u: GPUBindGroupEntry, b: RayBuffers, counter: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        layout: this.marchPipeline.getBindGroupLayout(0),
        entries: [
          u,
          { binding: 1, resource: { buffer: b.state } },
          { binding: 2, resource: { buffer: b.status } },
          { binding: 3, resource: { buffer: b.result } },
          { binding: 4, resource: { buffer: counter } },
        ],
      });
    const makeShade = (
      u: GPUBindGroupEntry,
      b: RayBuffers,
      accum: GPUBuffer,
      target: GPUTextureView,
      fallback: GPUTextureView,
    ): GPUBindGroup =>
      device.createBindGroup({
        layout: this.shadePipeline.getBindGroupLayout(0),
        entries: [
          u,
          { binding: 1, resource: { buffer: b.state } }, // t at disk crossing (M7 retardation)
          { binding: 2, resource: { buffer: b.status } },
          { binding: 3, resource: { buffer: b.result } },
          { binding: 5, resource: target },
          { binding: 6, resource: { buffer: this.ntSliceBuffer } },
          { binding: 7, resource: { buffer: this.planckBuffer } },
          { binding: 8, resource: fallback },
          { binding: 9, resource: this.fallbackSampler },
          { binding: 10, resource: { buffer: accum } },
          { binding: 11, resource: this.skyView },
          { binding: 12, resource: this.skySampler },
          { binding: 13, resource: this.skyAggView },
          { binding: 14, resource: { buffer: this.starBuffer } },
          { binding: 15, resource: { buffer: this.cellStartBuffer } },
        ],
      });

    // Chained fallbacks, coarsest first: level i falls back to level i-1
    // (the coarsest to the black dummy), full resolution to the finest level.
    // Every slab of a grid shares the grid's counter and fallback; a slab of
    // a split grid shades into its own target (composed by copy in encode).
    const wireGrid = (grid: SlabGrid, counter: GPUBuffer, target: GPUTextureView, fallback: GPUTextureView): void => {
      for (const slab of grid.slabs) {
        const u = { binding: 0, resource: { buffer: slab.uniformBuffer } };
        slab.initBindGroup = makeInit(u, slab.buffers);
        slab.marchBindGroup = makeMarch(u, slab.buffers, counter);
        slab.shadeBindGroup = makeShade(u, slab.buffers, slab.accum, slab.view ?? target, fallback);
      }
    };
    let coarserView = this.fallbackDummy.createView();
    for (const level of this.levels) {
      wireGrid(level.grid!, level.counter, level.view!, coarserView);
      coarserView = level.view!;
    }
    wireGrid(this.screenGrid!, this.counter, this.hdrView, coarserView);
  }

  /**
   * Compute the observer frame in float64 (src/physics) and upload uniforms.
   * `tile` (M7 export) renders this grid as a window of a larger image; the
   * default (0, 0, width, height) reproduces the untiled mapping bit-exactly
   * (init.wgsl). The frustum aspect always follows the FULL image dimensions.
   */
  setCamera(pose: CameraPose, spin = 0, observer: ObserverType = 'zamo', tile: TileSpec | null = null): void {
    this.camera = pose;
    this.spin = spin;
    this.observer = observer;

    const m = createMetric();
    evalMetric(spin, pose.r, pose.theta, m);
    let tetrad = orthonormalTetrad(m, observerFourVelocity(observer, m, spin, pose.r, pose.theta));
    tetrad = rotateTetradLook(tetrad, pose.yaw ?? 0, pose.pitch ?? 0);

    const fullW = tile ? tile.fullWidth : this.width;
    const fullH = tile ? tile.fullHeight : this.height;
    const tanHalfFovY = Math.tan(pose.fovY / 2);
    const tanHalfFovX = (tanHalfFovY * fullW) / fullH;
    const tolerance = QUALITY_TIERS[this.tier];

    const f32 = new Float32Array(this.uniformData);
    const u32 = new Uint32Array(this.uniformData);
    for (let leg = 0; leg < 4; leg++) f32.set(tetrad[leg]!, leg * 4);
    f32.set([m.g_tt, m.g_tph, m.g_rr, m.g_thth], 16);
    f32.set([m.g_phph, tanHalfFovX, tanHalfFovY, spin], 20);
    f32.set([pose.r, pose.theta, pose.phi, 0], 24);
    f32.set([tile ? tile.x : 0, tile ? tile.y : 0, fullW, fullH], 28);
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
    this.packDiskUniforms(f32);
    f32.set([0, 0, 1, 0], 48); // base sample: no jitter, overwrite weight
    f32.set([SKY_RADIANCE_SCALE, STAR_PSF_SIGMA_PX, this.disk.inhomogeneity, this.diskTime], 52);
    this.writeGridUniforms(this.screenGrid!, this.uniformData);

    // Preview levels: identical camera/physics words (same tan-half-FOVs, so
    // every grid spans the same frustum); only the pixel-grid words differ.
    for (const level of this.levels) {
      const levelData = this.uniformData.slice(0);
      new Uint32Array(levelData).set([level.width, level.height, level.width * level.height], 32);
      new Float32Array(levelData).set([0, 0, level.width, level.height], 28);
      this.writeGridUniforms(level.grid!, levelData);
      level.knownMarching = level.grid!.rayCount;
    }

    this.needsInit = true;
    this.knownMarching = this.screenGrid!.rayCount;
    this.generation++;
    this.epoch++;
    this.refineRound = 0;
    this.refineActive = false;
    this.refineMarching = 0;
    if (import.meta.env?.DEV) this.uniformHashAtRestart = hashUniforms(this.uniformData);
  }

  /** Dev-only invariant (IMPLEMENTATION.md M5 risk): uniforms may only change
   * through setCamera, which restarts the accumulation. A mismatch here means
   * some code path mutated uniforms without restarting — stale accumulation. */
  private uniformHashAtRestart = 0;

  /**
   * Upload per-slab uniform blocks derived from a grid's canonical block:
   * identical physics/camera/sample words everywhere; a slab overrides only
   * its tile-window and grid-size words — each slab is a tile of the
   * canonical image through the M7 tile path. A single-slab grid uploads the
   * canonical block unchanged (byte-identical to the pre-split uniforms).
   */
  private writeGridUniforms(grid: SlabGrid, canonical: ArrayBuffer): void {
    for (const slab of grid.slabs) {
      let data = canonical;
      if (grid.slabs.length > 1) {
        data = canonical.slice(0);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32[29] = f32[29]! + (slab.y0 - slab.apronTop); // tile.y of this band
        u32[33] = slab.gridH; // resolution.y
        u32[34] = grid.width * slab.gridH; // rayCount
      }
      this.gpu.device.queue.writeBuffer(slab.uniformBuffer, 0, data);
    }
  }

  /**
   * Disk + LUT uniform words (f32 indices 40-47). The T̂ slice at the current
   * spin is interpolated on the CPU in float64 grid space and uploaded, so the
   * GPU never interpolates across spins. Temperature and radiance scales:
   *   physical mode: T_obs = g·T̂·(Ṁc⁶/(GM)²σ)^{1/4}  (true kelvin)
   *   scaled mode:   the g = 1 peak is mapped to TEMP_SCALED_PEAK_K
   *                  (the documented global T rescale, PHYSICS.md §8)
   * The radiance scale maps the g = 1 peak XYZ luminance to HDR_PEAK_Y — a
   * global linear pre-exposure (camera model); per-pixel relative radiometry
   * is untouched.
   */
  private packDiskUniforms(f32: Float32Array): void {
    if (!this.disk.enabled || !this.luts) {
      f32.set([0, 0, 0, 0], 40);
      f32.set([0, 0, 0, 0], 44);
      this.ntSlice = null;
      this.diskShading = null;
      return;
    }
    const { nt, planck } = this.luts;
    const rIsco = iscoRadius(this.spin);
    const slice = sliceNtAtSpin(nt, this.spin);
    this.ntSlice = slice;
    this.gpu.device.queue.writeBuffer(this.ntSliceBuffer, 0, slice);

    let tHatPeak = 0;
    for (const v of slice) tHatPeak = Math.max(tHatPeak, v);
    const physScaleK = diskEffectiveTemperatureK(1, this.disk.massKg, this.disk.mdotKgPerS);
    const tempScaleK = this.disk.colorMode === 'scaled' ? TEMP_SCALED_PEAK_K / tHatPeak : physScaleK;
    const yPeak = samplePlanckXyz(planck, tHatPeak * tempScaleK)[1];
    const radianceScale = yPeak > 0 ? HDR_PEAK_Y / yPeak : 0;
    this.diskShading = { rIsco, rOuter: this.disk.rOuter, tempScaleK, radianceScale };
    f32.set([rIsco, this.disk.rOuter, tempScaleK, radianceScale], 40);
    f32.set([nt.uMax, 0, planck.log10TMin, planck.log10TMax], 44);
  }

  private diskShading: { rIsco: number; rOuter: number; tempScaleK: number; radianceScale: number } | null =
    null;

  /** The exact scale values uploaded to the shader (validation: CPU shade oracle). */
  getDiskShading(): { rIsco: number; rOuter: number; tempScaleK: number; radianceScale: number } | null {
    return this.disk.enabled && this.luts ? this.diskShading : null;
  }

  /** Swap disk parameters; restarts the accumulation like any physics change. */
  setDisk(params: DiskParams): void {
    this.disk = { ...params };
    if (this.camera) this.setCamera(this.camera, this.spin, this.observer);
  }

  getDisk(): DiskParams {
    return { ...this.disk };
  }

  /**
   * Advance the disk-inhomogeneity clock (M7): t_sim is a uniform word, so an
   * active pattern restarts the accumulation exactly like the circular-orbit
   * camera phase (ARCHITECTURE.md §6) — the preview ladder presents the
   * advection smoothly. A no-op whenever the pattern is off (axisymmetric
   * disks stay time-invariant: no restart, the M5-validated behavior).
   */
  setDiskTime(tSim: number): void {
    if (tSim === this.diskTime) return;
    this.diskTime = tSim;
    if (this.camera && this.disk.enabled && this.disk.inhomogeneity > 0 && this.luts) {
      this.setCamera(this.camera, this.spin, this.observer);
    }
  }

  /** Upload the production sky (M7, D9); restarts the accumulation. */
  setSky(data: SkyAssetData): void {
    const { device } = this.gpu;
    if (data.cellStart.length !== SKY_GRID_CELLS + 1) {
      throw new Error('sky assets: cell grid does not match SKY_GRID_CELLS (regenerate assets/sky).');
    }
    const upload = (label: string, faceSize: number, texels: Uint8Array): GPUTexture => {
      const tex = device.createTexture({
        label,
        size: { width: faceSize, height: faceSize, depthOrArrayLayers: 6 },
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const bytesPerFace = faceSize * faceSize * 8;
      for (let f = 0; f < 6; f++) {
        device.queue.writeTexture(
          { texture: tex, origin: { x: 0, y: 0, z: f } },
          texels.subarray(f * bytesPerFace, (f + 1) * bytesPerFace) as Uint8Array<ArrayBuffer>,
          { bytesPerRow: faceSize * 8 },
          { width: faceSize, height: faceSize },
        );
      }
      return tex;
    };
    this.skyTexture.destroy();
    this.skyAggTexture.destroy();
    this.starBuffer.destroy();
    this.cellStartBuffer.destroy();
    this.skyTexture = upload('sky-cubemap', data.faceSize, data.cubemap);
    this.skyView = this.skyTexture.createView({ dimension: 'cube' });
    this.skyAggTexture = upload('star-agg', data.aggFaceSize, data.aggCubemap);
    this.skyAggView = this.skyAggTexture.createView({ dimension: 'cube' });
    this.starBuffer = device.createBuffer({
      size: Math.max(32, data.starRecords.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'sky-stars',
    });
    device.queue.writeBuffer(this.starBuffer, 0, data.starRecords);
    this.cellStartBuffer = device.createBuffer({
      size: data.cellStart.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'sky-cellstart',
    });
    device.queue.writeBuffer(this.cellStartBuffer, 0, data.cellStart);
    this.buildBindGroups();
    if (this.camera) this.setCamera(this.camera, this.spin, this.observer);
  }

  /** Disable the interaction preview ladder (tiled export: only the converged
   * full-resolution result is read; previews would be dead work). */
  setPreviewsEnabled(on: boolean): void {
    this.previewsEnabled = on;
  }

  /** Provide the loaded LUTs (renderer bootstrap); enables disk shading. */
  setLuts(luts: DiskLuts): void {
    if (luts.nt.rCount * 4 !== this.ntSliceBuffer.size || luts.planck.data.byteLength !== this.planckBuffer.size) {
      throw new Error('LUT dimensions do not match the generator constants (regenerate assets/luts).');
    }
    this.luts = luts;
    this.gpu.device.queue.writeBuffer(this.planckBuffer, 0, luts.planck.data);
    if (this.camera) this.setCamera(this.camera, this.spin, this.observer);
  }

  getLuts(): DiskLuts | null {
    return this.luts;
  }

  /** Current-spin T̂ slice (validation harness: CPU shade oracle). */
  getNtSlice(): Float32Array | null {
    return this.ntSlice;
  }

  setQuality(tier: QualityTier): void {
    this.tier = tier;
    if (this.camera) this.setCamera(this.camera, this.spin, this.observer);
  }

  getCamera(): CameraPose | null {
    return this.camera;
  }

  getKnownMarching(): number {
    return this.camera ? this.knownMarching : 0;
  }

  /** Compose a split grid's slab targets into the grid target (interior rows
   * only — aprons are traced for Jacobian support, never presented). */
  private composeGrid(encoder: GPUCommandEncoder, grid: SlabGrid, target: GPUTexture): void {
    if (grid.slabs.length < 2) return;
    for (const slab of grid.slabs) {
      encoder.copyTextureToTexture(
        { texture: slab.texture!, origin: { x: 0, y: slab.apronTop } },
        { texture: target, origin: { x: 0, y: slab.y0 } },
        { width: grid.width, height: slab.rows },
      );
    }
  }

  /** Encode passes A-C for this frame. Returns false when no camera is set. */
  encode(encoder: GPUCommandEncoder): boolean {
    if (!this.camera || !this.screenGrid) return false;
    const groupsX = Math.ceil(this.width / WORKGROUP_2D);

    if (import.meta.env?.DEV && !this.needsInit && hashUniforms(this.uniformData) !== this.uniformHashAtRestart) {
      console.error('LensForge invariant violated: trace uniforms changed without an accumulation restart.');
    }

    // Pose changed this frame: every level re-inits. The full-resolution
    // march is deferred to the next frame — on an init frame its chunk would
    // be discarded by the next interaction re-init before any ray it advances
    // can terminate and be presented, so the frame budget goes to the preview
    // ladder (which IS presented). At rest this costs exactly one frame.
    const didInit = this.needsInit;
    if (this.needsInit) {
      const pass = encoder.beginComputePass({
        label: 'trace-init',
        timestampWrites: this.profiler.timestampWrites('init') as GPUComputePassTimestampWrites,
      });
      pass.setPipeline(this.initPipeline);
      for (const slab of this.screenGrid.slabs) {
        pass.setBindGroup(0, slab.initBindGroup);
        pass.dispatchWorkgroups(groupsX, Math.ceil(slab.gridH / WORKGROUP_2D));
      }
      if (this.previewsEnabled) {
        for (const level of this.levels) {
          for (const slab of level.grid!.slabs) {
            pass.setBindGroup(0, slab.initBindGroup);
            pass.dispatchWorkgroups(Math.ceil(level.width / WORKGROUP_2D), Math.ceil(slab.gridH / WORKGROUP_2D));
          }
          level.knownMarching = level.grid!.rayCount;
        }
      }
      pass.end();
      this.needsInit = false;
      this.knownMarching = this.screenGrid.rayCount;
    }

    // Preview ladder, coarsest first (each level's shade may sample the
    // coarser level shaded earlier in this same encoder). Chunks are encoded
    // back-to-back — terminated rays exit each chunk immediately. Only the
    // final chunk's marching count is read back, so the counter is cleared
    // once, just before it.
    for (const level of this.previewsEnabled ? this.levels : []) {
      const active = level.knownMarching > 0;
      const slabs = level.grid!.slabs;
      if (active) {
        const pass = encoder.beginComputePass({
          label: `preview${level.config.scale}-march`,
          timestampWrites: this.profiler.timestampWrites(
            `preview${level.config.scale}`,
          ) as GPUComputePassTimestampWrites,
        });
        pass.setPipeline(this.marchPipeline);
        const chunks = level.config.governed
          ? Math.min(
              PREVIEW_CHUNKS_MAX,
              Math.max(PREVIEW_CHUNKS_MIN, Math.round(level.config.dispatchesPerFrame * this.workScale)),
            )
          : level.config.dispatchesPerFrame;
        for (let d = 0; d < chunks - 1; d++) {
          for (const slab of slabs) {
            pass.setBindGroup(0, slab.marchBindGroup);
            pass.dispatchWorkgroups(Math.ceil(slab.buffers.rayCount / this.marchWorkgroup));
          }
        }
        pass.end();
        encoder.clearBuffer(level.counter);
        const last = encoder.beginComputePass({ label: `preview${level.config.scale}-march-final` });
        last.setPipeline(this.marchPipeline);
        for (const slab of slabs) {
          last.setBindGroup(0, slab.marchBindGroup);
          last.dispatchWorkgroups(Math.ceil(slab.buffers.rayCount / this.marchWorkgroup));
        }
        last.end();
        if (!level.mapPending) {
          encoder.copyBufferToBuffer(level.counter, 0, level.counterStaging, 0, 4);
          level.copyEncoded = true;
        }
      }
      if (active || didInit) {
        const pass = encoder.beginComputePass({ label: `preview${level.config.scale}-shade` });
        pass.setPipeline(this.shadePipeline);
        for (const slab of slabs) {
          pass.setBindGroup(0, slab.shadeBindGroup);
          pass.dispatchWorkgroups(Math.ceil(level.width / WORKGROUP_2D), Math.ceil(slab.gridH / WORKGROUP_2D));
        }
        pass.end();
        this.composeGrid(encoder, level.grid!, level.texture!);
      }
    }

    // Full-resolution march ladder — on interaction frames too: rays that
    // terminate within it present at full resolution immediately, so sharp
    // detail survives motion wherever geodesics resolve quickly (most of the
    // frame); the preview ladder covers the rest. Chunk count is governed.
    // Once the base trace has converged, the same ladder drives temporal
    // refinement rounds (REFINE_MAX_ROUNDS): re-init with sub-pixel jitter,
    // march across frames, fold the converged round into the accumulation.
    const baseMarching = this.knownMarching > 0;
    let roundComplete = false;
    if (!baseMarching && this.refinementEnabled && !this.refineActive && this.refineRound < REFINE_MAX_ROUNDS) {
      this.startRefineRound(encoder, groupsX);
    } else if (!baseMarching && this.refineActive && this.refineMarching === 0) {
      // The jittered round has converged: one accumulation shade folds it in.
      roundComplete = true;
      this.refineActive = false;
      this.refineRound++;
    }

    if (baseMarching || (this.refineActive && this.refineMarching > 0)) {
      const slabs = this.screenGrid.slabs;
      // Export/batch mode (previews disabled): no frame budget to honor —
      // run the maximum chunk ladder per frame (each dispatch stays bounded
      // at STEPS_PER_DISPATCH attempts, so V-P4 TDR safety is unchanged).
      const chunks = this.previewsEnabled
        ? Math.min(MAIN_CHUNKS_MAX, Math.max(1, Math.round(MAIN_CHUNKS_BASE * this.workScale)))
        : MAIN_CHUNKS_MAX;
      if (chunks > 1) {
        const pass = encoder.beginComputePass({
          label: 'trace-march',
          timestampWrites: this.profiler.timestampWrites('march') as GPUComputePassTimestampWrites,
        });
        pass.setPipeline(this.marchPipeline);
        for (let d = 0; d < chunks - 1; d++) {
          for (const slab of slabs) {
            pass.setBindGroup(0, slab.marchBindGroup);
            pass.dispatchWorkgroups(Math.ceil(slab.buffers.rayCount / this.marchWorkgroup));
          }
        }
        pass.end();
      }
      encoder.clearBuffer(this.counter);
      const last = encoder.beginComputePass({
        label: 'trace-march-final',
        timestampWrites: (chunks > 1
          ? undefined
          : this.profiler.timestampWrites('march')) as GPUComputePassTimestampWrites,
      });
      last.setPipeline(this.marchPipeline);
      for (const slab of slabs) {
        last.setBindGroup(0, slab.marchBindGroup);
        last.dispatchWorkgroups(Math.ceil(slab.buffers.rayCount / this.marchWorkgroup));
      }
      last.end();
      if (!this.counterMapPending) {
        encoder.copyBufferToBuffer(this.counter, 0, this.counterStaging, 0, 4);
        this.counterCopyEncoded = true;
      }
    }

    // Full-resolution shade runs only when it is meaningful AND idempotent-
    // or-intended: every frame of the base trace (weight 1 — overwrite),
    // and exactly once per completed refinement round (weight 1/(n+1) —
    // re-running it would fold the same sample in twice and bias the mean).
    // Mid-round frames skip it: statuses are MARCHING for the jittered rays
    // and the presented image stays the last completed accumulation.
    if (baseMarching || didInit || roundComplete) {
      const pass = encoder.beginComputePass({
        label: 'trace-shade',
        timestampWrites: this.profiler.timestampWrites('shade') as GPUComputePassTimestampWrites,
      });
      pass.setPipeline(this.shadePipeline);
      for (const slab of this.screenGrid.slabs) {
        pass.setBindGroup(0, slab.shadeBindGroup);
        pass.dispatchWorkgroups(groupsX, Math.ceil(slab.gridH / WORKGROUP_2D));
      }
      pass.end();
      this.composeGrid(encoder, this.screenGrid, this.hdrTexture);
    }
    return true;
  }

  /** Radical-inverse (Halton) low-discrepancy sequence for sample jitter. */
  private static halton(index: number, base: number): number {
    let f = 1;
    let r = 0;
    let i = index;
    while (i > 0) {
      f /= base;
      r += f * (i % base);
      i = Math.floor(i / base);
    }
    return r;
  }

  /** Begin a jittered refinement round: sanctioned uniform update (jitter +
   * accumulation weight; the dev invariant hash is refreshed) and a
   * full-resolution re-init. Physics words are untouched. */
  private startRefineRound(encoder: GPUCommandEncoder, groupsX: number): void {
    const f32 = new Float32Array(this.uniformData);
    f32[48] = TracePass.halton(this.refineRound + 1, 2) - 0.5;
    f32[49] = TracePass.halton(this.refineRound + 1, 3) - 0.5;
    f32[50] = 1 / (this.refineRound + 2); // base sample + completed rounds + this one
    this.writeGridUniforms(this.screenGrid!, this.uniformData);
    if (import.meta.env?.DEV) this.uniformHashAtRestart = hashUniforms(this.uniformData);

    const pass = encoder.beginComputePass({ label: 'refine-init' });
    pass.setPipeline(this.initPipeline);
    for (const slab of this.screenGrid!.slabs) {
      pass.setBindGroup(0, slab.initBindGroup);
      pass.dispatchWorkgroups(groupsX, Math.ceil(slab.gridH / WORKGROUP_2D));
    }
    pass.end();
    this.refineMarching = this.screenGrid!.rayCount;
    this.refineActive = true;
    this.epoch++;
  }

  /**
   * Toggle temporal refinement (validation harness: measurements read the
   * base-sample statuses/HDR, which refinement rounds would re-init under
   * them). Turning it off mid-refinement restarts the base trace so the
   * persistent ray state is the exact pixel-center sample again.
   */
  setRefinementEnabled(on: boolean): void {
    if (this.refinementEnabled === on) return;
    this.refinementEnabled = on;
    if (!on && (this.refineActive || this.refineRound > 0) && this.camera) {
      this.setCamera(this.camera, this.spin, this.observer);
    }
  }

  /** Accumulated sample count (profiler HUD). */
  getRefinementState(): { samples: number; maxSamples: number; refining: boolean } {
    return {
      samples: this.refineRound + 1,
      maxSamples: REFINE_MAX_ROUNDS + 1,
      refining: this.refineActive,
    };
  }

  /** Feedback control on measured total trace GPU time (see GOVERNOR_*). */
  private workScale = 1;

  private updateGovernor(): void {
    let total = 0;
    let sampled = false;
    for (const name of ['init', 'preview4', 'preview2', 'march', 'shade']) {
      const ms = this.profiler.emaMs(name);
      if (ms !== null) {
        total += ms;
        sampled = true;
      }
    }
    if (!sampled || total <= 0) return; // no timestamp-query: baselines apply
    const step = total > GOVERNOR_TARGET_TRACE_MS ? GOVERNOR_STEP_DOWN : GOVERNOR_STEP_UP;
    this.workScale = Math.min(GOVERNOR_SCALE_MAX, Math.max(GOVERNOR_SCALE_MIN, this.workScale * step));
  }

  /** Governor telemetry (profiler HUD). */
  getGovernorState(): { workScale: number; targetMs: number } {
    return { workScale: this.workScale, targetMs: GOVERNOR_TARGET_TRACE_MS };
  }

  /** Kick off the async still-marching counter readbacks; call once per frame after submit. */
  afterSubmit(): void {
    this.updateGovernor();
    const generationAtRead = this.generation;
    const epochAtRead = this.epoch;
    const isRefine = this.refineActive && this.knownMarching === 0;
    if (this.counterCopyEncoded && !this.counterMapPending && (this.knownMarching > 0 || isRefine)) {
      this.counterCopyEncoded = false;
      this.counterMapPending = true;
      void this.counterStaging
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const count = new Uint32Array(this.counterStaging.getMappedRange())[0]!;
          this.counterStaging.unmap();
          if (epochAtRead !== this.epoch) return;
          if (isRefine) this.refineMarching = count;
          else this.knownMarching = count;
        })
        .catch(() => {
          /* device loss: recovery rebuilds this pass */
        })
        .finally(() => {
          this.counterMapPending = false;
        });
    }
    for (const level of this.levels) {
      if (!level.copyEncoded || level.mapPending || level.knownMarching === 0) continue;
      level.copyEncoded = false;
      level.mapPending = true;
      void level.counterStaging
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const count = new Uint32Array(level.counterStaging.getMappedRange())[0]!;
          level.counterStaging.unmap();
          if (generationAtRead === this.generation) level.knownMarching = count;
        })
        .catch(() => {
          /* device loss: recovery rebuilds this pass */
        })
        .finally(() => {
          level.mapPending = false;
        });
    }
  }

  /** Read back the screen status plane (validation: shadow-diameter
   * measurement). Slab-split grids are assembled from the interior rows of
   * each slab — the returned full-image plane is identical to a monolithic
   * readback (aprons are duplicated tiles of the same rays). */
  async readScreenStatuses(): Promise<Uint32Array> {
    const grid = this.screenGrid;
    if (!grid) throw new Error('trace pass has no screen buffers');
    const out = new Uint32Array(grid.width * grid.height);
    for (const slab of grid.slabs) {
      const raw = new Uint32Array((await this.readBuffer(slab.buffers.status)).buffer);
      out.set(raw.subarray(slab.apronTop * grid.width, (slab.apronTop + slab.rows) * grid.width), slab.y0 * grid.width);
    }
    return out;
  }

  /** Read back the screen result planes (validation: star-splat Jacobian
   * spot-check — ESCAPED rays hold their asymptotic direction here). Returned
   * plane-major over the full image, assembled from slab interiors. */
  async readScreenResults(): Promise<Float32Array> {
    const grid = this.screenGrid;
    if (!grid) throw new Error('trace pass has no screen buffers');
    const pixels = grid.width * grid.height;
    const out = new Float32Array(RESULT_PLANES * pixels);
    for (const slab of grid.slabs) {
      const raw = new Float32Array((await this.readBuffer(slab.buffers.result)).buffer);
      const slabRays = slab.buffers.rayCount;
      for (let p = 0; p < RESULT_PLANES; p++) {
        out.set(
          raw.subarray(p * slabRays + slab.apronTop * grid.width, p * slabRays + (slab.apronTop + slab.rows) * grid.width),
          p * pixels + slab.y0 * grid.width,
        );
      }
    }
    return out;
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
  async runBatch(
    initialStates: Float64Array,
    spin: number,
    maxDispatches = 400,
    disk: { rIsco: number; rOuter: number } | null = null,
  ): Promise<BatchResult> {
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
    // Batch disk geometry (V-O2); shading scales are irrelevant — Pass B only.
    f32.set(disk ? [disk.rIsco, disk.rOuter, 0, 0] : [0, 0, 0, 0], 40);
    f32.set([0, 0, 0, 0], 28); // deterministic tile words (Pass B ignores them)
    f32.set([0, 0, 1, 0], 48); // deterministic base-sample words (Pass B ignores them)
    f32.set([0, 0, 0, 0], 52); // deterministic sky words (Pass B ignores them)
    const batchUniforms = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'batch-uniforms',
    });
    device.queue.writeBuffer(batchUniforms, 0, uniformData);

    // A batch owns its counter: the live loop's counter may be in continuous
    // use by refinement rounds while a validation batch runs (M6).
    const batchCounter = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'batch-marching-counter',
    });
    const bindGroup = device.createBindGroup({
      layout: this.marchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: batchUniforms } },
        { binding: 1, resource: { buffer: buffers.state } },
        { binding: 2, resource: { buffer: buffers.status } },
        { binding: 3, resource: { buffer: buffers.result } },
        { binding: 4, resource: { buffer: batchCounter } },
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
      encoder.clearBuffer(batchCounter);
      const pass = encoder.beginComputePass({ label: 'batch-march' });
      pass.setPipeline(this.marchPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(rayCount / this.marchWorkgroup));
      pass.end();
      encoder.copyBufferToBuffer(batchCounter, 0, counterRead, 0, 4);
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
    batchCounter.destroy();
    counterRead.destroy();
    return { statuses: statusBack, states, directions, steps, dispatches };
  }

  destroy(): void {
    if (this.screenGrid) this.destroyGrid(this.screenGrid);
    for (const level of this.levels) {
      if (level.grid) this.destroyGrid(level.grid);
      level.texture?.destroy();
      level.counter.destroy();
      level.counterStaging.destroy();
    }
    this.counter.destroy();
    this.counterStaging.destroy();
    this.fallbackDummy.destroy();
    this.ntSliceBuffer.destroy();
    this.planckBuffer.destroy();
    this.skyTexture.destroy();
    this.skyAggTexture.destroy();
    this.starBuffer.destroy();
    this.cellStartBuffer.destroy();
  }
}
