# Author: Nathaniel Ruhl
# This script determines the out-of plane angle during the crossing

from Modules.OrbitModel import OrbitModel

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd


# This function calculates the out-of plane angle during the time range of the horizon crossing
# Inputs: observation dictionary
# Outputs: returns the mean out-of-plane angle in radians
def calculate_psi(obs_dict):
    fn_string = obs_dict["rossi_path"]
    time_range_crossing = obs_dict["crossing_time_range"]
    s_unit = obs_dict['starECI']

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

    return np.mean(psi_crossing)