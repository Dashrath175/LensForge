# LensForge — Physics

All equations, conventions, numerical methods, and accuracy boundaries. Implemented twice: float64 in `src/physics/` (oracle), float32 in `src/shaders/` (D3–D5 in ARCHITECTURE.md govern the formulation choices). Citations `[R#]` → REFERENCES.md.

Signature `(−,+,+,+)`. Greek indices 0–3, coordinates `x^μ = (t, r, θ, φ)`.

## 1. Units

Geometrized units `G = c = 1`; additionally `M = 1` inside all kernels (D5). Spin `a ≡ J/M ∈ [0, 1)`. Radii are in units of `GM/c²`. Physical mass and accretion rate enter only through UI-side conversions in `src/core/`:

- Length: `r_phys = r · GM/c²`, time: `t_phys = t · GM/c³`
- Disk temperature scale: `T_eff ∝ Ṁ^{1/4} M^{−1/2}` (§7)

## 2. Kerr Metric (Boyer–Lindquist)

With `Σ = r² + a²cos²θ`, `Δ = r² − 2r + a²`, `A = (r² + a²)² − a²Δ sin²θ`:

```
ds² = −(1 − 2r/Σ) dt² − (4ar sin²θ/Σ) dt dφ + (Σ/Δ) dr² + Σ dθ² + (A/Σ) sin²θ dφ²
```

Inverse metric (all the integrator needs, D3):

```
g^tt = −A/(ΣΔ)        g^tφ = −2ar/(ΣΔ)      g^rr = Δ/Σ
g^θθ = 1/Σ            g^φφ = (Δ − a²sin²θ)/(ΣΔ sin²θ)
```

Horizons `r± = 1 ± √(1 − a²)`; ergosphere `r_E(θ) = 1 + √(1 − a²cos²θ)`. Schwarzschild is `a = 0` throughout — there is no separate Schwarzschild implementation (MASTER_PLAN.md §5); its closed forms serve as test oracles (§8, VALIDATION.md V-A).

## 3. Analytic Benchmarks (used only for validation)

| Quantity | Expression | Source |
|---|---|---|
| Schwarzschild photon sphere | `r_ph = 3` | [R9] |
| Schwarzschild critical impact parameter | `b_c = 3√3 ≈ 5.19615` | [R9] |
| Weak-field deflection | `α = 4/b + 15π/(4b²) + O(b⁻³)` | [R8] |
| Kerr equatorial photon orbits | `r_ph = 2[1 + cos(⅔ arccos(∓a))]` (− prograde) | [R10] |
| ISCO | `Z₁ = 1 + (1−a²)^{1/3}[(1+a)^{1/3} + (1−a)^{1/3}]`, `Z₂ = √(3a² + Z₁²)`, `r_isco = 3 + Z₂ ∓ √((3−Z₁)(3+Z₁+2Z₂))` (− prograde) | [R10] |
| Kerr shadow contour | Bardeen's `(α(r), β(r))` parametric curve over unstable photon orbits | [R1] |

## 4. Geodesics — Hamiltonian Formulation

Super-Hamiltonian `H = ½ g^{μν} p_μ p_ν`; photons satisfy `H = 0`. Equations of motion in affine parameter `λ`:

```
dx^μ/dλ = g^{μν} p_ν
dp_μ/dλ = −½ (∂_μ g^{αβ}) p_α p_β        (only ∂_r and ∂_θ are nonzero)
```

Eight ODEs; requires only the inverse metric and its `r`, `θ` gradients — no Christoffel symbols, no square-root sign tracking (D3). Conserved along rays and used **only as error monitors** (never integrated):

```
E = −p_t      L_z = p_φ      Q = p_θ² + cos²θ (L_z²/sin²θ − a²E²)      H = 0
```

## 5. Observers, Tetrads, Ray Initialization

Camera is a local orthonormal tetrad `{e_0 = u, e_1, e_2, e_3}` for the chosen observer:

- **Static observer** (`u ∝ ∂_t`): valid outside the ergosphere only.
- **ZAMO** (`u ∝ ∂_t + ω ∂_φ`, `ω = −g_tφ/g_φφ = 2ar/A`): valid everywhere outside the horizon; default.
- **Circular-orbit observer**: `u^μ = u^t (1, 0, 0, Ω)` (§6); gives orbital-camera aberration exactly.

Pixel `(i,j)` → unit direction `n̂` in the tetrad frame via pinhole FOV → `k^μ = e_0^μ + n^i e_i^μ` → `p_μ = g_{μν} k^ν`. Relativistic aberration is fully contained in the tetrad boost — it must never be implemented separately.

## 6. Disk Kinematics

Prograde circular equatorial orbits (emitters), `r ≥ r_isco`:

```
Ω(r) = 1/(r^{3/2} + a)
u^t  = (r^{3/2} + a) / [ r^{3/4} √( r^{3/2} − 3 r^{1/2} + 2a ) ]
u^μ  = u^t (1, 0, 0, Ω)
```

[R10]. Below the ISCO the disk is truncated (zero-torque boundary, §7); plunging-region emission is out of scope (§10).

## 7. Disk Emission — Novikov–Thorne

Time-averaged, geometrically thin, optically thick, equatorial disk [R4]. Local flux `F(r)` from Page & Thorne [R3] (their Eq. 15n with the relativistic structure function); precomputed by `tools/` into a `(r, a)` LUT — it involves an integral with no closed form for `a ≠ 0`. Effective temperature:

```
T_eff(r) = [ F(r) / σ_SB ]^{1/4}
```

Sanity limit (`a = 0`, Newtonian regime): `F(r) → (3Ṁ/8πr³)(1 − C/√r)` with `C = √6 + (√3/2)ln((√6+√3)/(√6−√3)) ≈ 3.976`; the naive Shakura–Sunyaev constant `√6` [R21] is approached only to leading order (the zero-torque log terms shift the boundary constant — V-A6 verifies both statements). Local emission is blackbody `B_ν(T_eff)` (no limb darkening; §9).

## 8. Radiative Transfer, Redshift, Beaming

`I_ν/ν³` is Lorentz- and GR-invariant along rays [R20]. Redshift factor between emitter and observer:

```
g = ν_obs/ν_em = (k_μ u^μ)_obs / (k_μ u^μ)_em = 1 / [ u^t (1 − Ω λ) ]     with λ = L_z/E
```

(compact form for circular emitters and a distant static observer [R6]). Then:

- Specific intensity: `I_ν,obs = g³ I_{ν/g, em}`
- Bolometric: `I_obs = g⁴ I_em`
- **Blackbody shortcut** (exact): `g³ B_ν(ν/g, T) = B_ν(ν, gT)` — a blackbody stays a blackbody with `T_obs = g·T_eff`. Shading therefore evaluates the Planck→CIE XYZ LUT directly at `g·T_eff(r)`; gravitational redshift, Doppler shift, and beaming are all inside `g` and are inseparable by construction.

Escaped rays sample the celestial sphere: surface brightness conservation makes the cubemap lookup through the deflected direction exact for extended sources; point stars additionally require the magnification `|det J|⁻¹` from the ray-bundle Jacobian (D9).

Color: Planck spectrum → CIE 1931 XYZ (1-D LUT in `T`) → linear sRGB [R17]. Two calibrations, both labeled in-UI: **physical** (true `T_eff`; UV-peaked disks render white-blue) and **temperature-scaled** (global rescale of `T` into the visible Planckian locus — the convention of every published visualization including DNGR [R5], made explicit here).

## 9. Assumptions

1. Geometric-optics limit (wavelength ≪ all curvature scales); no wave effects.
2. Vacuum propagation outside the disk — no plasma dispersion or scattering.
3. Disk: geometrically thin, optically thick, equatorial, time-averaged axisymmetric, zero torque at ISCO, local blackbody emission, isotropic in the local rest frame (no limb darkening).
4. Higher-order *lensed images* of the disk are included (they come free from the integrator); disk self-irradiation (returning radiation heating the disk) is not.
5. Camera: ideal pinhole; exposure/tone-mapping/optional bloom form an explicit, documented "camera model" — the only non-physical layer (MASTER_PLAN.md §8).
6. Optional disk inhomogeneity (M7) is a statistical brightness field advected at the exact `Ω(r)`; its transport is physical, its texture is a labeled statistical model standing in for GRMHD turbulence.

## 10. Numerical Methods & Accuracy Boundaries

**Integrator:** Dormand–Prince RK5(4) [R12] with embedded error estimate, adaptive step in `λ`, PI step-size controller [R11]. Absolute + relative tolerance is the single quality knob (D11). Step clamped near the horizon (`h ∝ r − r₊`) and near the equatorial plane when the disk is enabled.

**Disk intersection:** detect sign change of `cos θ` across a step with `r ∈ [r_isco, r_out]`; refine by 3–4 bisection/Hermite iterations on dense output. Never rely on a step landing on the plane.

**Termination:** `r < r₊(1 + ε)` (`ε ≈ 10⁻³`) → CAPTURED (redshift has already driven any contribution below one ULP of the framebuffer — this is why Boyer–Lindquist is safe, D4); outward crossing of `r_esc ≈ 10⁴` → ESCAPED, with the state Hermite-interpolated to **exactly** `r = r_esc` before the asymptotic direction is extracted (the sky is sampled at a well-defined radius, and the f32/f64 implementations compare at the same radius — otherwise the differing overshoot radii alone contribute ~`4/r_esc` of spurious deviation); step budget exhausted → flagged and counted (VALIDATION.md V-O3).

**Monitors:** per-step drift in `E`, `L_z`, `Q` and `|H|`; a step exceeding the monitor threshold is rejected and halved.

**Float32 boundaries (WGSL has no f64):**

| Issue | Consequence | Mitigation |
|---|---|---|
| ~7 significant digits | Phase error accumulates over photon-ring winding | `M = 1` units (D5); monitors reject bad steps |
| Cancellation in `Δ` near `r₊` | Step-size collapse near horizon | Termination at `r₊(1+ε)`; clamped steps |
| Exponential impact-parameter sensitivity of subrings [R7] | Photon subrings `n ≥ 2` unresolvable in real time | Documented scope boundary (MASTER_PLAN.md §4); df64 offline mode is future work |

Direct (`n = 0`) and first photon-ring (`n = 1`) images are quantitatively trustworthy; the V-O test suite measures exactly how trustworthy, per quality tier.
