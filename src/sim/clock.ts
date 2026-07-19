/**
 * Simulation clock (ARCHITECTURE.md §6): the sim loop advances coordinate
 * time t_sim (geometric units, GM/c³) by dt_real × timeScale each tick —
 * microseconds of CPU work, no physics stepping. The render loop never owns
 * time; it only reads parameters whose evolution t_sim drives (circular-orbit
 * camera phase now; disk inhomogeneity phase in M7).
 */
export class SimClock {
  private tSimValue = 0;

  /** Advance by a real-time interval scaled into simulation time. */
  advance(dtRealSeconds: number, timeScale: number): number {
    const dtSim = dtRealSeconds * timeScale;
    this.tSimValue += dtSim;
    return dtSim;
  }

  get tSim(): number {
    return this.tSimValue;
  }

  reset(): void {
    this.tSimValue = 0;
  }
}
