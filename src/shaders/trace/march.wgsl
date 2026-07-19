// Pass B — chunked geodesic march (ARCHITECTURE.md §5, D6): each dispatch
// advances every MARCHING ray by at most stepsPerDispatch DP45 *attempts*
// (accepted or rejected), bounding worst-case dispatch duration regardless of
// step-controller behaviour (V-P4). Float32 port of integrateGeodesic.

#include "../common/uniforms.wgsl"
#include "../common/ray-state.wgsl"
#include "../integrator/dp45.wgsl"
#include "../disk/kinematics.wgsl"

override WORKGROUP_SIZE: u32 = 64u;
override MAX_TOTAL_STEPS: f32 = 8192.0;

// Max |dTheta| per step while inside the disk's radial band, so equatorial
// plane crossings cannot be skipped — twin of THETA_STEP_CAP in
// src/physics/integrator.ts (PHYSICS.md §10).
const DISK_THETA_STEP_CAP: f32 = 0.15;

// Bisect a sign change of cos(theta(sigma)) on the Hermite dense output over
// [lo, hi] — port of bisectSignChange + cosThetaAt in src/physics/integrator.ts.
// Returns the crossing sigma, or -1.0 when the endpoint signs agree.
fn diskCrossingSigma(
  y0: array<f32, 8>,
  f0: array<f32, 8>,
  y1: array<f32, 8>,
  f1: array<f32, 8>,
  h: f32,
  sigmaLo: f32,
  sigmaHi: f32,
) -> f32 {
  var lo = sigmaLo;
  var hi = sigmaHi;
  var cLo = cos(hermiteTheta(y0, f0, y1, f1, h, lo));
  let cHi = cos(hermiteTheta(y0, f0, y1, f1, h, hi));
  if (cLo == 0.0) { return lo; }
  if (cHi == 0.0) { return hi; }
  if (sign(cLo) == sign(cHi)) { return -1.0; }
  for (var iter = 0; iter < 26; iter++) {
    let mid = 0.5 * (lo + hi);
    let cMid = cos(hermiteTheta(y0, f0, y1, f1, h, mid));
    if (sign(cMid) == sign(cLo)) {
      lo = mid;
      cLo = cMid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

@group(0) @binding(4) var<storage, read_write> marchingCount: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_march(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.rayCount || status[i] != STATUS_MARCHING) {
    return;
  }

  let a = u.camMetricB.w;
  var y = loadRayState(i);
  var h = state[P_H * u.rayCount + i];
  var errOld = state[P_ERR_OLD * u.rayCount + i];
  var steps = state[P_STEPS * u.rayCount + i];

  var k1 = geodesicRhs(a, y);
  if (h == 0.0) {
    h = initialStepSize(y, k1, u.rtol, u.atol);
  }

  var st = STATUS_MARCHING;
  for (var attempt = 0u; attempt < u.stepsPerDispatch; attempt++) {
    // Termination checks on the current accepted state (PHYSICS.md §10).
    if (y[IR] <= u.rCapture) {
      st = STATUS_CAPTURED;
      break;
    }
    if (y[IR] >= u.rEscape && k1[IR] > 0.0) {
      // Only reachable for rays initialized at/beyond the escape radius; the
      // normal path is the interpolated crossing in the accept branch below.
      st = STATUS_ESCAPED;
      break;
    }
    if (steps >= MAX_TOTAL_STEPS) {
      st = STATUS_MAX_STEPS;
      break;
    }

    // Horizon clamp: no stage may evaluate below the capture radius.
    if (k1[IR] < 0.0) {
      let drAbs = abs(k1[IR]);
      if (drAbs > 0.0) {
        h = min(h, max(0.5 * (y[IR] - u.rCapture), 1e-7) / drAbs);
      }
    }
    // Disk-band theta step cap — twin of src/physics/integrator.ts.
    if (u.disk.y > 0.0 && y[IR] > 0.5 * u.disk.x && y[IR] < 2.0 * u.disk.y) {
      let dthAbsDisk = abs(k1[ITH]);
      if (dthAbsDisk > 0.0) {
        h = min(h, DISK_THETA_STEP_CAP / dthAbsDisk);
      }
    }
    // Axis-whip clamp — twin of src/physics/integrator.ts (PHYSICS.md §10):
    // never straddle the 1/sin(theta) boundary layer of a polar turning point.
    let pphClamp = y[IPPH];
    if (pphClamp != 0.0) {
      let sAxis = abs(sin(y[ITH]));
      if (sAxis < 0.5) {
        let lOverS = pphClamp / sAxis;
        let P = sqrt(y[IPTH] * y[IPTH] + lOverS * lOverS);
        let sMin = abs(pphClamp) / max(P, 1e-30);
        let dthAbs = abs(k1[ITH]);
        if (dthAbs > 0.0) {
          h = min(h, max(0.5 * sAxis, 0.25 * sMin) / dthAbs);
        }
      }
    }

    let att = dp45Attempt(a, y, k1, h, u.rtol, u.atol);
    if (att.err <= 1.0) {
      // Disk plane crossing on the accepted step (PHYSICS.md §10): both
      // half-intervals are checked so an enter-and-exit double crossing within
      // one step cannot be missed; crossings outside [rIsco, rOuter] pass
      // through — literal port of src/physics/integrator.ts. Higher-order
      // lensed disk images need no special handling: rays that miss the band
      // keep marching and may cross the plane again after wrapping.
      if (u.disk.y > 0.0) {
        var hitFound = false;
        for (var bracket = 0; bracket < 2 && !hitFound; bracket++) {
          let lo = 0.5 * f32(bracket);
          let sigma = diskCrossingSigma(y, k1, att.yTrial, att.k7, h, lo, lo + 0.5);
          if (sigma >= 0.0) {
            let yHit = hermiteState(y, k1, att.yTrial, att.k7, h, sigma);
            if (yHit[IR] >= u.disk.x && yHit[IR] <= u.disk.y) {
              y = yHit;
              steps += 1.0;
              st = STATUS_HIT_DISK;
              hitFound = true;
            }
          }
        }
        if (hitFound) {
          break;
        }
      }

      // Escape-radius crossing: interpolate to exactly r = rEscape so the sky
      // is sampled at a well-defined radius, matching the oracle (V-O1).
      if (y[IR] < u.rEscape && att.yTrial[IR] >= u.rEscape) {
        var lo = 0.0;
        var hi = 1.0;
        for (var iter = 0; iter < 30; iter++) {
          let mid = 0.5 * (lo + hi);
          let yMid = hermiteState(y, k1, att.yTrial, att.k7, h, mid);
          if (yMid[IR] < u.rEscape) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        y = hermiteState(y, k1, att.yTrial, att.k7, h, 0.5 * (lo + hi));
        steps += 1.0;
        st = STATUS_ESCAPED;
        break;
      }
      y = att.yTrial;
      k1 = att.k7; // FSAL
      steps += 1.0;
      let scale = clamp(SAFETY * pow(att.err, -PI_ALPHA) * pow(errOld, PI_BETA), MIN_SCALE, MAX_SCALE);
      h *= scale;
      errOld = max(att.err, 1e-6);
    } else {
      h *= max(MIN_SCALE, SAFETY * pow(att.err, -0.2));
    }
  }

  storeRayState(i, y);
  state[P_H * u.rayCount + i] = h;
  state[P_ERR_OLD * u.rayCount + i] = errOld;
  state[P_STEPS * u.rayCount + i] = steps;

  if (st == STATUS_ESCAPED) {
    let m = evalInverseMetric(a, y[IR], y[ITH]);
    // g^phiphi p_phi is exactly zero for axial rays (p_phi = 0) — same
    // axis-regular treatment as geodesicRhs (src/physics/integrator.ts twin).
    let dphi = select(m.gphph * y[IPPH], 0.0, y[IPPH] == 0.0);
    let dir = cartesianDirection(
      a, y[IR], y[ITH], y[IPH],
      m.grr * y[IPR], m.gthth * y[IPTH], m.gtph * y[IPT] + dphi,
    );
    result[0u * u.rayCount + i] = dir.x;
    result[1u * u.rayCount + i] = dir.y;
    result[2u * u.rayCount + i] = dir.z;
  }

  if (st == STATUS_HIT_DISK) {
    // Payload for Pass C: (r, phi, g) of the refined crossing. g uses the
    // exact contraction for the circular emitter (shaders/disk/kinematics.wgsl).
    result[0u * u.rayCount + i] = y[IR];
    result[1u * u.rayCount + i] = y[IPH];
    result[2u * u.rayCount + i] = diskRedshift(a, y[IR], -y[IPT], y[IPPH]);
  }

  status[i] = st;
  if (st == STATUS_MARCHING) {
    atomicAdd(&marchingCount, 1u);
  }
}
