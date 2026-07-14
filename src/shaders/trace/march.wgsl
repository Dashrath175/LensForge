// Pass B — chunked geodesic march (ARCHITECTURE.md §5, D6): each dispatch
// advances every MARCHING ray by at most stepsPerDispatch DP45 *attempts*
// (accepted or rejected), bounding worst-case dispatch duration regardless of
// step-controller behaviour (V-P4). Float32 port of integrateGeodesic.

#include "../common/uniforms.wgsl"
#include "../common/ray-state.wgsl"
#include "../integrator/dp45.wgsl"

override WORKGROUP_SIZE: u32 = 64u;
override MAX_TOTAL_STEPS: f32 = 8192.0;

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

    let att = dp45Attempt(a, y, k1, h, u.rtol, u.atol);
    if (att.err <= 1.0) {
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
    let dir = cartesianDirection(
      a, y[IR], y[ITH], y[IPH],
      m.grr * y[IPR], m.gthth * y[IPTH], m.gtph * y[IPT] + m.gphph * y[IPPH],
    );
    result[0u * u.rayCount + i] = dir.x;
    result[1u * u.rayCount + i] = dir.y;
    result[2u * u.rayCount + i] = dir.z;
  }

  status[i] = st;
  if (st == STATUS_MARCHING) {
    atomicAdd(&marchingCount, 1u);
  }
}
