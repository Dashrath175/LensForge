/**
 * M5 ParameterStore: restart/continue classification (ARCHITECTURE.md §6)
 * and observer validity guards (IMPLEMENTATION.md M5).
 */
import { describe, expect, it } from 'vitest';
import { DEFAULT_PARAMETERS } from '../../../src/core/parameters';
import { ergosphereRadius } from '../../../src/physics/metric';
import { iscoRadius } from '../../../src/physics/orbits';
import { ParameterStore } from '../../../src/sim/parameterStore';

describe('change classification (restart vs continue)', () => {
  it('classifies physics-affecting changes as physics', () => {
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    expect(store.update({ spin: 0.5 }).kind).toBe('physics');
    expect(store.update({ camera: { phi: 1 } }).kind).toBe('physics');
    expect(store.update({ observer: 'static' }).kind).toBe('physics');
    expect(store.update({ disk: { rOuter: 25 } }).kind).toBe('physics');
    expect(store.update({ quality: 'medium' }).kind).toBe('physics');
  });

  it('classifies exposure as display and timeScale as time — no restart', () => {
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    expect(store.update({ exposureStops: 1.5 }).kind).toBe('display');
    expect(store.update({ timeScale: 100 }).kind).toBe('time');
  });

  it('no-op updates and camera-mode-only changes never signal physics', () => {
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    expect(store.update({}).kind).toBe('none');
    expect(store.update({ spin: store.get().spin }).kind).toBe('none');
    expect(store.update({ camera: { mode: 'look' } }).kind).toBe('none');
  });

  it('notifies subscribers exactly on observable changes', () => {
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    let calls = 0;
    store.subscribe(() => calls++);
    store.update({ spin: 0.3 });
    store.update({ spin: 0.3 }); // identical — no notification
    expect(calls).toBe(1);
  });
});

describe('observer validity guards', () => {
  it('static observer stays valid across the reachable domain (R_MIN > max ergosphere)', () => {
    // The ergosphere never exceeds 2M (equator); R_MIN = 4 keeps the static
    // observer valid everywhere today. The ZAMO fallback is defense in depth
    // for future sub-ergosphere cameras (free-fall observers, §9).
    expect(ergosphereRadius(0.998, Math.PI / 2)).toBeLessThan(4);
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    store.update({ spin: 0.998, observer: 'static' });
    const change = store.update({ camera: { r: 1 } }); // clamped to R_MIN = 4
    expect(store.get().camera.r).toBe(4);
    expect(store.get().observer).toBe('static');
    expect(change.warning).toBeNull();
  });

  it('circular observer is coerced to the equator and clamped to the ISCO', () => {
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    store.update({ spin: 0 });
    const change = store.update({ observer: 'circular', camera: { r: 4, theta: 1.0 } });
    expect(store.get().camera.theta).toBe(Math.PI / 2);
    expect(store.get().camera.r).toBeCloseTo(iscoRadius(0), 12); // 6M
    expect(change.warning).toMatch(/ISCO/);
  });

  it('raising spin keeps a circular observer on a stable orbit', () => {
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    store.update({ observer: 'circular', camera: { r: 6 } });
    expect(store.get().camera.r).toBeCloseTo(6, 12);
    store.update({ spin: 0.9 });
    // ISCO(0.9) ≈ 2.32 < 6: unchanged.
    expect(store.get().camera.r).toBeCloseTo(6, 12);
  });

  it('range clamps hold (spin cap, pole exclusion, fov bounds)', () => {
    const store = new ParameterStore(DEFAULT_PARAMETERS);
    store.update({ spin: 2 });
    expect(store.get().spin).toBe(0.998);
    store.update({ camera: { theta: -1 } });
    expect(store.get().camera.theta).toBeGreaterThan(0);
    store.update({ camera: { fovY: 99 } });
    expect(store.get().camera.fovY).toBeLessThanOrEqual(2.0);
  });
});
