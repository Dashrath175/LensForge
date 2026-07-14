/**
 * Physical constants (SI, CODATA 2018) used only for unit conversion at the
 * UI/tools boundary. All kernel physics is in geometrized units G = c = M = 1
 * (PHYSICS.md §1, D5) and must not import these.
 */
export const GRAVITATIONAL_CONSTANT = 6.6743e-11; // m^3 kg^-1 s^-2
export const SPEED_OF_LIGHT = 2.99792458e8; // m s^-1
export const STEFAN_BOLTZMANN = 5.670374419e-8; // W m^-2 K^-4
export const PLANCK_CONSTANT = 6.62607015e-34; // J s
export const BOLTZMANN_CONSTANT = 1.380649e-23; // J K^-1
export const SOLAR_MASS = 1.98892e30; // kg
