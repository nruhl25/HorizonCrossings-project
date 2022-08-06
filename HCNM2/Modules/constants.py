# This file contains gravitational constants and constants for the orbited planet

G = 6.6743*10**(-11)     # Nm^2/kg^2, Gravitational constant
R_EARTH = 6378.107     # km, equatorial radius... average volumetric radius of earth is 6371
M_EARTH = 5.972 * 10 ** 24   # kg, known mass of Earth
MU = G * M_EARTH    # Nm^2/kg, Earth's gravitational parameter

# Oblate Earth Model - WGS84 datum
a = b = 6378.137  # [km] semi-major axes
c = 6356.752  # [km] semi-minor axis
e = 0.08182   # Eccentricity from Bate et al.
