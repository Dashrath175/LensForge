# Release media

Presentation renders used by the top-level README. Every image is real
application output — rendered by the release build via the dev-console
hooks, converged (base sample + temporal-AA refinement), and captured
through the standard present transform (AgX). No external editing beyond the
typography composited onto `banner.png`.

| File | Contents | Provenance |
|---|---|---|
| `banner.png` | Repository banner | *DNGR Gargantua* preset at 2560×860, `fovY = 0.62`, `yaw = −0.28`, exposure +1.2 stops; wordmark composited |
| `orbit-gargantua.png` | Animated PNG, 24 frames, full 2π camera orbit | *DNGR Gargantua* preset at 480×270, exposure +1.5 stops; each frame fully converged |
| `luminet-1979.png` | *Luminet 1979* preset | 1920×1080, preset defaults |
| `dngr-gargantua.png` | *DNGR Gargantua* preset | 1920×1080, exposure +1.5 stops (display-only lift; the beamed disk is extremely high-contrast at `a = 0.998`) |
| `face-on.png` | *Face-on* preset | 1920×1080, preset defaults |
| `photon-ring-grazing.png` | *Photon-ring grazing* preset | 1920×1080, preset defaults |
| `default-view.png` | Startup configuration | 1920×1080 (edge-on Schwarzschild, disk turbulence 1.0, bloom 0.10) |
| `luminet-physical-color.png` | *Luminet 1979* with physical color mode | 1920×1080 (true blackbody chromaticity at `T_obs = g·T_eff`) |

Exposure is a display-only parameter (present pass); it never changes the
underlying radiometry. The scientifically authoritative renders remain the
golden images in [`../figures/presets/`](../figures/presets/) — those are
bit-validated against `tests/golden/` by V-I5 and are not presentation
media.
