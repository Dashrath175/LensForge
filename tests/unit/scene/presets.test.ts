/**
 * M5 preset validation (IMPLEMENTATION.md M5): byte round-trip and framing
 * reproduction — shipped presets ARE the V-I reference framings.
 */
import { describe, expect, it } from 'vitest';
import { criticalImpactParameter } from '../../../src/physics/shadow';
import {
  parsePreset,
  PRESET_VERSION,
  serializePreset,
  SHIPPED_PRESETS,
  VI1_FRAMING,
  VI3_FRAMING,
} from '../../../src/scene/presets';

describe('preset serialization', () => {
  it('save → load → save is byte-identical for every shipped preset', () => {
    for (const preset of SHIPPED_PRESETS) {
      const first = serializePreset(preset);
      const second = serializePreset(parsePreset(first));
      expect(second, preset.name).toBe(first);
    }
  });

  it('round-trips a mutated full state exactly (JSON numbers are lossless)', () => {
    const p = structuredClone(SHIPPED_PRESETS[1]!);
    p.params.camera.phi = Math.PI / 7;
    p.params.camera.yaw = -0.0123456789012345;
    p.params.disk.mdotKgPerS = 10 ** 21.15;
    const json = serializePreset(p);
    const back = parsePreset(json);
    expect(back).toEqual(p);
    expect(serializePreset(back)).toBe(json);
  });

  it('rejects malformed presets', () => {
    expect(() => parsePreset('{}')).toThrow(/version/);
    expect(() => parsePreset(JSON.stringify({ version: 1, name: 'x' }))).toThrow(/params/);
    const good = JSON.parse(serializePreset(SHIPPED_PRESETS[0]!)) as {
      params: { camera: Record<string, unknown> };
    };
    delete good.params.camera['yaw'];
    expect(() => parsePreset(JSON.stringify(good))).toThrow(/yaw/);
  });
});

describe('shipped presets reproduce reference framings', () => {
  it('Luminet 1979 uses the V-I3 framing at a = 0', () => {
    const p = SHIPPED_PRESETS.find((x) => x.name === 'Luminet 1979')!;
    expect(p.params.spin).toBe(0);
    expect(p.params.camera.r).toBe(VI3_FRAMING.r);
    expect(p.params.camera.theta).toBe(VI3_FRAMING.theta);
    expect(p.params.camera.fovY).toBe(VI3_FRAMING.fovY);
    expect(p.params.camera.yaw).toBe(0);
    expect(p.params.disk.enabled).toBe(true);
  });

  it('Photon-ring grazing centers the prograde critical curve via Bardeen b_c', () => {
    const p = SHIPPED_PRESETS.find((x) => x.name === 'Photon-ring grazing')!;
    expect(p.params.camera.r).toBe(VI1_FRAMING.r);
    expect(p.params.camera.yaw).toBeCloseTo(criticalImpactParameter(0.9, true) / VI1_FRAMING.r, 15);
    expect(p.params.camera.fovY).toBeLessThan(0.01);
  });

  it('all shipped presets carry the full parameter state', () => {
    for (const p of SHIPPED_PRESETS) {
      expect(p.version).toBe(PRESET_VERSION);
      // parsePreset enforces completeness; a shipped preset must survive it.
      expect(() => parsePreset(serializePreset(p))).not.toThrow();
    }
  });

  it('accepts version-1 presets with the M7 fields defaulted to 0', () => {
    // A v1 preset is a v2 preset minus disk.inhomogeneity and bloomStrength.
    const v2 = JSON.parse(serializePreset(SHIPPED_PRESETS[0]!)) as {
      version: number;
      params: { disk: Record<string, unknown>; bloomStrength?: number };
    };
    v2.version = 1;
    delete v2.params.disk['inhomogeneity'];
    delete v2.params.bloomStrength;
    const parsed = parsePreset(JSON.stringify(v2));
    expect(parsed.version).toBe(PRESET_VERSION);
    expect(parsed.params.disk.inhomogeneity).toBe(0);
    expect(parsed.params.bloomStrength).toBe(0);
  });
});
