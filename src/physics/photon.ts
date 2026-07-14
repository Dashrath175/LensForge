/**
 * PHYSICS.md §5 — photon initial conditions.
 *
 * Two constructors: from an observer tetrad (the camera path), and directly
 * from conserved quantities (the validation path — exact impact parameters
 * with no tetrad involved, so the two constructions cross-check each other).
 */
import { IPH, IPR, IPT, IPTH, IPPH, IR, IT, ITH, STATE_SIZE } from './hamiltonian';
import { createMetric, evalMetric, type MetricBL } from './metric';
import type { FourVector, Tetrad } from './tetrad';

/**
 * Photon state from a tetrad and a unit spatial direction n̂ in the local
 * frame: k = e_0 + n^i e_i, then p_μ = g_μν k^ν. Null by construction.
 */
export function photonFromTetrad(
  m: MetricBL,
  tetrad: Tetrad,
  n: [number, number, number],
  position: { t?: number; r: number; theta: number; phi?: number },
): Float64Array {
  const norm = Math.hypot(n[0], n[1], n[2]);
  const k: FourVector = [0, 0, 0, 0];
  for (let mu = 0; mu < 4; mu++) {
    k[mu] =
      tetrad[0][mu]! +
      (n[0] / norm) * tetrad[1][mu]! +
      (n[1] / norm) * tetrad[2][mu]! +
      (n[2] / norm) * tetrad[3][mu]!;
  }
  const y = new Float64Array(STATE_SIZE);
  y[IT] = position.t ?? 0;
  y[IR] = position.r;
  y[ITH] = position.theta;
  y[IPH] = position.phi ?? 0;
  y[IPT] = m.g_tt * k[0]! + m.g_tph * k[3]!;
  y[IPR] = m.g_rr * k[1]!;
  y[IPTH] = m.g_thth * k[2]!;
  y[IPPH] = m.g_tph * k[0]! + m.g_phph * k[3]!;
  return y;
}

/**
 * Equatorial photon at radius r with E = 1, L_z = b (exact impact parameter),
 * p_θ = 0; p_r solved from H = 0. `ingoing` selects the p_r sign.
 */
export function equatorialPhoton(a: number, r: number, b: number, ingoing: boolean): Float64Array {
  const m = evalMetric(a, r, Math.PI / 2, createMetric());
  const pt = -1;
  const pph = b;
  const radicand = -(m.gtt * pt * pt + 2 * m.gtph * pt * pph + m.gphph * pph * pph) / m.grr;
  if (radicand < 0) {
    throw new RangeError(`No equatorial photon with b=${b} at r=${r} (inside radial turning point).`);
  }
  const y = new Float64Array(STATE_SIZE);
  y[IR] = r;
  y[ITH] = Math.PI / 2;
  y[IPT] = pt;
  y[IPR] = (ingoing ? -1 : 1) * Math.sqrt(radicand);
  y[IPPH] = pph;
  return y;
}
