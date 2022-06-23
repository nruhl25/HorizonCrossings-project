# Author: Nathaniel Ruhl
# This script determines the out-of plane angle during the crossing

from Modules.OrbitModel import OrbitModel
from Modules import tools

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd


# This function calculates and plots an array of out-of plane angles during the time range of the horizon crossing
def calculate_psi_crossing(fn_string, time_range_crossing, s_unit):
    r_array, v_array, t_array = OrbitModel.read_rxte_orbit(fn_string)
    h_array = np.cross(r_array, v_array)
    h_mags = np.linalg.norm(h_array, axis=1)
    h_mags_matrix = np.column_stack((h_mags, h_mags, h_mags))
    h_array = h_array / h_mags_matrix   # unit pole vector

    # Will get 3 or 4 data points in the time range
    crossing_indices = np.where((t_array >= time_range_crossing[0]) & (t_array <= time_range_crossing[1]))[0]
    h = h_array[crossing_indices]
    psi_crossing = np.zeros_like(h)
    for i in range(len(crossing_indices)):
        psi_crossing[i] = np.pi/2 - np.arccos(np.dot(h[i], s_unit))  # out-of plane angle array, rad

    return np.rad2deg(psi_crossing), t_array[crossing_indices]


if __name__ == "__main__":
    # Inputs for a specific observation
    s_unit = tools.celestial_to_geocentric(np.deg2rad(83.63426), np.deg2rad(22.010))
    time_range_crossing = np.array([300, 600])+2.06414e8
    fn_rossi = cwd + "/Data/RXTE/50098/FPorbit_Day2389"

    psi_crossing, time_crossing = calculate_psi_crossing(fn_rossi, time_range_crossing, s_unit)

    plt.plot(time_crossing, psi_crossing, '.')
    plt.xlabel("time")
    plt.ylabel("out-of-plane angle (deg)")
    plt.show()

