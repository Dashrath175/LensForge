/**
 * WebGPU device acquisition, capability tiering, and device-lost recovery (D1, D6).
 *
 * Device loss is a first-class event on Windows (TDR watchdog): the owner
 * passes `onDeviceLost`, and the app layer rebuilds all GPU state through a
 * fresh `acquireGpu` call. Nothing below the app layer caches a GPUDevice
 * across a loss.
 */

export type GpuTier = 'high' | 'medium' | 'low';

export interface GpuContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly canvasFormat: GPUTextureFormat;
  readonly hasTimestampQuery: boolean;
  readonly tier: GpuTier;
}

export class WebGpuUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGpuUnavailableError';
  }
}

/**
 * Placeholder tiering until M6 introduces the startup micro-benchmark
 * (IMPLEMENTATION.md M6): discrete-class limits => high, otherwise medium.
 */
function classifyTier(adapter: GPUAdapter): GpuTier {
  const l = adapter.limits;
  if (l.maxComputeInvocationsPerWorkgroup >= 1024 && l.maxStorageBufferBindingSize >= 1 << 30) {
    return 'high';
  }
  if (l.maxComputeInvocationsPerWorkgroup >= 256) return 'medium';
  return 'low';
}

export async function acquireGpu(
  canvas: HTMLCanvasElement,
  onDeviceLost: (info: GPUDeviceLostInfo) => void,
): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new WebGpuUnavailableError('navigator.gpu is not available in this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new WebGpuUnavailableError('No WebGPU adapter available (GPU may be blocklisted).');
  }

  const hasTimestampQuery = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestampQuery ? ['timestamp-query'] : [],
    label: 'lensforge-device',
  });

  // `lost` resolves exactly once per device; 'destroyed' means we tore it down deliberately.
  void device.lost.then((info) => {
    if (info.reason !== 'destroyed') onDeviceLost(info);
  });

  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new WebGpuUnavailableError('Failed to create a WebGPU canvas context.');
  }
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });

  return { adapter, device, context, canvasFormat, hasTimestampQuery, tier: classifyTier(adapter) };
}
