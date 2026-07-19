/**
 * GPU pass timing via timestamp queries (ARCHITECTURE.md §5). Ground truth
 * for all performance work (DEVELOPMENT.md §6). Degrades to a no-op when the
 * 'timestamp-query' feature is absent, so callers never branch.
 */

const MAX_PASSES = 16;

interface PassSlot {
  /** Exponential moving average of GPU duration, milliseconds. */
  emaMs: number;
  /** Insertion order, for stable report layout. */
  order: number;
}

export class GpuProfiler {
  private readonly device: GPUDevice;
  private readonly enabled: boolean;
  private readonly querySet: GPUQuerySet | null = null;
  private readonly resolveBuffer: GPUBuffer | null = null;
  private readonly readBuffer: GPUBuffer | null = null;
  /** EMAs keyed by pass name — pass sets differ frame to frame (e.g. march
   * dispatches stop once converged), so slot-by-order would blend passes. */
  private readonly slots = new Map<string, PassSlot>();
  /** Pass names of this frame's timestamp pairs, in query-index order. */
  private framePassNames: string[] = [];
  private frameCursor = 0;
  private readPending = false;

  constructor(device: GPUDevice, enabled: boolean) {
    this.device = device;
    this.enabled = enabled;
    if (!enabled) return;
    this.querySet = device.createQuerySet({
      type: 'timestamp',
      count: MAX_PASSES * 2,
      label: 'profiler-timestamps',
    });
    this.resolveBuffer = device.createBuffer({
      size: MAX_PASSES * 2 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      label: 'profiler-resolve',
    });
    this.readBuffer = device.createBuffer({
      size: MAX_PASSES * 2 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'profiler-read',
    });
  }

  /** Call when encoding a pass; returns timestampWrites for the pass descriptor (or undefined). */
  timestampWrites(name: string): GPUComputePassTimestampWrites | GPURenderPassTimestampWrites | undefined {
    if (!this.enabled || !this.querySet || this.frameCursor >= MAX_PASSES) return undefined;
    const slotIndex = this.frameCursor++;
    this.framePassNames[slotIndex] = name;
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: slotIndex * 2,
      endOfPassWriteIndex: slotIndex * 2 + 1,
    };
  }

  /** Encode resolution of this frame's queries; call once per frame after all passes. */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.enabled || !this.querySet || !this.resolveBuffer || !this.readBuffer) return;
    if (this.frameCursor === 0) return;
    encoder.resolveQuerySet(this.querySet, 0, this.frameCursor * 2, this.resolveBuffer, 0);
    if (!this.readPending) {
      encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, this.frameCursor * 2 * 8);
    }
  }

  /** Kick off an async read of resolved timestamps; safe to call every frame. */
  endFrame(): void {
    const passCount = this.frameCursor;
    const passNames = this.framePassNames;
    this.frameCursor = 0;
    this.framePassNames = [];
    if (!this.enabled || !this.readBuffer || this.readPending || passCount === 0) return;
    this.readPending = true;
    void this.readBuffer
      .mapAsync(GPUMapMode.READ, 0, passCount * 2 * 8)
      .then(() => {
        const buf = this.readBuffer;
        if (!buf) return;
        const ts = new BigInt64Array(buf.getMappedRange(0, passCount * 2 * 8));
        for (let i = 0; i < passCount; i++) {
          const name = passNames[i];
          if (!name) continue;
          const dtMs = Number(ts[i * 2 + 1]! - ts[i * 2]!) * 1e-6;
          if (dtMs < 0 || dtMs >= 10_000) continue;
          const slot = this.slots.get(name);
          if (!slot) this.slots.set(name, { emaMs: dtMs, order: this.slots.size });
          else slot.emaMs = slot.emaMs * 0.9 + dtMs * 0.1;
        }
        buf.unmap();
      })
      .catch(() => {
        /* device loss mid-read: HUD simply stops updating; recovery rebuilds the profiler */
      })
      .finally(() => {
        this.readPending = false;
      });
  }

  report(): string {
    if (!this.enabled) return 'gpu timings: timestamp-query unavailable';
    return [...this.slots.entries()]
      .sort((a, b) => a[1].order - b[1].order)
      .map(([name, s]) => `${name}: ${s.emaMs.toFixed(3)} ms`)
      .join('\n');
  }

  /** Smoothed GPU duration of a named pass, ms — or null before first sample
   * (or without timestamp-query). Drives the frame-budget governor. */
  emaMs(name: string): number | null {
    return this.slots.get(name)?.emaMs ?? null;
  }
}
