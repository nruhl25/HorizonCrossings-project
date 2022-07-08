# Author: Nathaniel Ruhl
# This script models the the maximum error resulting from making the geocentric approximation, rather than using the geodetic approximation

# For earth, the maximum difference is 75 m at 45 degrees latitude (ref. Clynch 2008 geodesy transforms)

import numpy as np
import matplotlib.pyplot as plt

from psi_solver_ellipsoid import point_on_earth_azimuth_polar

R_planet = np.linalg.norm(point_on_earth_azimuth_polar(0, np.pi/4))
max_tp_err = 75e-3  # km

dr_list = []
R_orbit_list = R_planet + np.arange(300, 20000, 50)

for R_orbit in R_orbit_list:
    Delta_theta = abs(np.arccos((R_planet+max_tp_err)/R_orbit) - np.arccos(R_planet/R_orbit))
    dr = R_orbit*Delta_theta*1000.    # in-track position error [m]
    dr_list.append(dr)

plt.plot(R_orbit_list - R_planet, dr_list)
plt.xlabel("Orbital altitude (km)")
plt.ylabel("Maximum in-track error caused by geocentric assumption (m)")
plt.show()
