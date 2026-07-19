# LensForge — Release Validation Record (v1.0.0, M8)

The auditable scientific record for the v1.0.0 release (VALIDATION.md §3:
"Comparison overlays are committed to `docs/` at each release"). Everything
here was measured on the release code by the `src/validation` harness driven
through the dev-build `__lensforge` hooks; raw measurement JSON is under
[`results/m8/`](results/m8/), figures under [`figures/`](figures/).
(Presentation renders for the top-level README live in [`media/`](media/README.md)
and are not part of the validation record.)

Hardware note: all GPU numbers in this record were measured on the
development GPU, not the reference RTX 4070 Ti named by V-P (VALIDATION.md
§4). Correctness results (V-O, V-I) are hardware-independent up to the
documented LSB-level FMA differences (D12); the V-P1 60 fps claim on
reference hardware remains inherited from the M6 record and is flagged in
Known Limitations.

## Figures

| Figure | Test | Result |
|---|---|---|
| [vi1-shadow-contours.svg](figures/vi1-shadow-contours.svg) | V-I1 — rendered shadow silhouette vs Bardeen analytic contour, `a ∈ {0, 0.5, 0.9, 0.998}` | max deviation ≤ **0.50 px** at all four spins (bound 1 px); zero MAX_STEPS pixels |
| [vo1-deviation.svg](figures/vo1-deviation.svg) | V-O1 — GPU f32 vs CPU f64 escape-direction deviation curves | High tier **0.052′ / 0.159′ / 0.99′** at `a = 0 / 0.9 / 0.998` (bound 1′); Medium 1.83′ (3′), Low 2.51′ (10′) |
| [vo2-disk-deviation.svg](figures/vo2-disk-deviation.svg) | V-O2 — disk-hit redshift-factor deviation | `Δg/g ≤ 1.03e−4` worst spin (bound 1e−3); normalized `Δr ≤ 2.9e−3` |
| [vi2-luminet-geometry.svg](figures/vi2-luminet-geometry.svg) | V-I2 — disk-image boundary geometry at the Luminet [R2] framing | 166/166 boundary transitions matched (0 fragile), max **0.62 px** absolute, 0.66 % shape deviation where resolvable |
| [vi3-beaming.svg](figures/vi3-beaming.svg) | V-I3 — beaming asymmetry vs float64 oracle | ratio agreement 2.1e−5 (`a = 0`), 5.7e−5 (`a = 0.9`) — bound 5 % |
| [presets/](figures/presets/) | Converged renders of all four shipped presets (displayed orientation — these are also the V-I5 goldens) | — |

## What each comparison actually establishes

- **V-I1** is a true literature comparison: the Bardeen critical curve [R1]
  is closed-form, mapped to pixels through an exact float64 tetrad inversion
  at finite camera radius (no far-field approximation), and confronted with
  the GPU-rendered silhouette.
- **V-I2** compares the GPU image against the float64 oracle at the
  project's Luminet framing (Schwarzschild, disk to 20M, viewed 10° above
  the plane — the [R2] photograph geometry). Luminet's figure is itself the numerical
  solution of these same equations; the oracle — independently validated
  against closed forms in V-A3/V-A4/V-A6 — stands in for the published curves
  at full precision. A digitization of the 1979 scanned figure is *not*
  performed; this is recorded as a limitation in VALIDATION.md §7.
- **V-I3** re-shades every sampled disk pixel through the float64 pipeline
  (intersection → g → Page–Thorne flux → Planck → XYZ) and compares total and
  per-pixel luminance left/right of the hole. The DNGR [R5] comparison is
  feature-level (approaching side, D-shadow, ring structure), not
  pixel-level: DNGR's frames include artistic grading that is out of scope.
- **V-I5** goldens are the four preset PNGs (SSIM > 0.99 gate) plus
  pre-tonemap HDR luminance statistics (< 1 % gate) in
  `tests/golden/baselines.json`. M8 measured SSIM = 1.0 and HDR mean error
  = 0 against the M7-captured statistics (bit-repeatable base sample on the
  capture hardware).

## Reproducing

```bash
npm run dev                     # then in the browser console (dev build):
await __lensforge.runValidation(0.998)        # V-O1/V-O4
await __lensforge.runDiskValidation(0.9)      # V-O2
await __lensforge.measureShadowContour(0.998) # V-I1
await __lensforge.runDiskEdgeValidation()     # V-I2
await __lensforge.measureBeaming(0.9)         # V-I3
await __lensforge.runStatusCensus()           # V-O3
await __lensforge.runGoldenSuite()            # V-I5
npm run tools:figures -- <capturesDir>        # rebuild figures from captures
```
