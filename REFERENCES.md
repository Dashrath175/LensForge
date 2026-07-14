# LensForge — References

Canonical citation keys `[R#]` used across all documentation. When adding an equation to PHYSICS.md or a comparison to VALIDATION.md, add or reuse a key here — never cite inline elsewhere.

## Research Papers — Black Hole Imaging & Geodesics

| Key | Reference | Used for |
|---|---|---|
| R1 | Bardeen, J. M. (1973). "Timelike and null geodesics in the Kerr metric." In *Black Holes (Les Astres Occlus)*, eds. DeWitt & DeWitt, Gordon and Breach, p. 215. | Analytic Kerr shadow contour (V-I1); photon orbit structure |
| R2 | Luminet, J.-P. (1979). "Image of a spherical black hole with thin accretion disk." *Astronomy & Astrophysics* 75, 228–235. | First correct disk image; isophote comparison (V-I2) |
| R5 | James, O., von Tunzelmann, E., Franklin, P., Thorne, K. S. (2015). "Gravitational lensing by spinning black holes in astrophysics, and in the movie *Interstellar*." *Classical and Quantum Gravity* 32, 065001. | DNGR renderer; high-spin disk morphology comparison (V-I3); documents the beaming-off artistic choice LensForge rejects |
| R6 | Cunningham, C. T. (1975). "The effects of redshifts and focusing on the spectrum of an accretion disk around a Kerr black hole." *Astrophysical Journal* 202, 788–802. | Transfer-function formalism; compact redshift factor `g = 1/[u^t(1 − Ωλ)]` (PHYSICS.md §8) |
| R7 | Gralla, S. E., Holz, D. E., Wald, R. M. (2019). "Black hole shadows, photon rings, and lensing rings." *Physical Review D* 100, 024018. | Photon subring exponential structure; justifies the `n ≥ 2` float32 scope boundary (PHYSICS.md §10) |
| R10 | Bardeen, J. M., Press, W. H., Teukolsky, S. A. (1972). "Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation." *Astrophysical Journal* 178, 347–370. | ISCO closed form, circular-orbit kinematics `Ω(r)`, `u^t` (PHYSICS.md §3, §6); ZAMO frames (§5) |

## Research Papers — Accretion Disk Physics

| Key | Reference | Used for |
|---|---|---|
| R3 | Page, D. N., Thorne, K. S. (1974). "Disk-accretion onto a black hole. Time-averaged structure of accretion disk." *Astrophysical Journal* 191, 499–506. | Relativistic flux profile `F(r)` with zero-torque ISCO boundary — the NT flux LUT source (PHYSICS.md §7, V-A6) |
| R4 | Novikov, I. D., Thorne, K. S. (1973). "Astrophysics of black holes." In *Black Holes (Les Astres Occlus)*, eds. DeWitt & DeWitt, Gordon and Breach, p. 343. | Thin-disk model foundations (PHYSICS.md §7) |
| R21 | Shakura, N. I., Sunyaev, R. A. (1973). "Black holes in binary systems. Observational appearance." *Astronomy & Astrophysics* 24, 337–355. | Newtonian disk limit used as V-A6 sanity check |

## Books — Relativity & Radiative Processes

| Key | Reference | Used for |
|---|---|---|
| R8 | Misner, C. W., Thorne, K. S., Wheeler, J. A. (1973). *Gravitation*. W. H. Freeman. | Conventions, tetrads, weak-field deflection series (PHYSICS.md §3, §5) |
| R9 | Chandrasekhar, S. (1983). *The Mathematical Theory of Black Holes*. Oxford University Press. | Kerr geodesic theory; Schwarzschild photon sphere and `b_c` (PHYSICS.md §3) |
| R20 | Rybicki, G. B., Lightman, A. P. (1979). *Radiative Processes in Astrophysics*. Wiley. | `I_ν/ν³` invariance and radiative transfer (PHYSICS.md §8) |

## Mathematical / Numerical References

| Key | Reference | Used for |
|---|---|---|
| R11 | Hairer, E., Nørsett, S. P., Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Springer. | Adaptive step control, PI controller, dense output (PHYSICS.md §10) |
| R12 | Dormand, J. R., Prince, P. J. (1980). "A family of embedded Runge-Kutta formulae." *Journal of Computational and Applied Mathematics* 6(1), 19–26. | DP45 coefficients (PHYSICS.md §10) |
| R15 | Thall, A. (2006). "Extended-precision floating-point numbers for GPU computation." SIGGRAPH '06 Research Posters. (Builds on Dekker, T. J. (1971), *Numerische Mathematik* 18, 224–242.) | df64 double-float technique — future offline mode (ARCHITECTURE.md §9) |
| R17 | Wyszecki, G., Stiles, W. S. (1982). *Color Science: Concepts and Methods, Quantitative Data and Formulae*, 2nd ed. Wiley. | Planck spectrum → CIE 1931 XYZ pipeline (PHYSICS.md §8) |
| R22 | Wyman, C., Sloan, P.-P., Shirley, P. (2013). "Simple Analytic Approximations to the CIE XYZ Color Matching Functions." *Journal of Computer Graphics Techniques* 2(2), 1–11. | Multi-lobe Gaussian CMF fits used by the Planck→XYZ LUT generator (`tools/`) |

## GPU / WebGPU / Rendering References

| Key | Reference | Used for |
|---|---|---|
| R13 | W3C. *WebGPU* specification. https://www.w3.org/TR/webgpu/ | Device, pipeline, timestamp-query semantics (`src/gpu`) |
| R14 | W3C. *WebGPU Shading Language (WGSL)* specification. https://www.w3.org/TR/WGSL/ | Kernel language semantics; float behavior underlying D12 |
| R16 | Sobotka, T. *AgX* picture formation. https://github.com/sobotka/AgX | Tone mapping choice (D10) |

## Data Sources

| Key | Reference | Used for |
|---|---|---|
| R18 | ESA/Gaia/DPAC. *Gaia DR3 all-sky star map*. https://sci.esa.int/gaia — License: CC BY-SA 3.0 IGO. | Celestial sphere HDR panorama + bright-star catalog (M7); requires in-app attribution |
| R19 | NASA/Goddard Space Flight Center Scientific Visualization Studio. *Deep Star Maps*. https://svs.gsfc.nasa.gov/4851 — Public domain. | Alternative sky panorama (M7) |
