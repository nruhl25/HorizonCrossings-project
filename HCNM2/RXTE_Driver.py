# Author: Nathaniel Ruhl
# This is the driver for RXTE analysis. We will create an example obs_dict

from Modules.OrbitModel import OrbitModel
from Modules.LocateR0hc import LocateR0hc
from Modules.TransmitModel import TransmitModel
from Modules.CurveComparison import CurveComparison

# import observation dictionary
from ObservationDictionaries.RXTE.test_obs import test_obs
from ObservationDictionaries.RXTE.dict_60079 import dict_60079

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[0])  # HCNM2/ is cwd

obs_dict = dict_60079
e_band = [4.0, 6.0]  # keV
bin_size = 1.0

# # 1) Define orbit model
r_array, v_array, t_array = OrbitModel.define_orbit_model(obs_dict, "rossi", time_step=0.01)

# 2) LocateR0hc (must know hc_type here, R_orbit and h_unit defined within the class)
r0_obj = LocateR0hc(obs_dict, r_array, v_array, t_array)
t0_model_index, lat_gp, lon_gp = r0_obj.return_r0_data()
del r0_obj

v0 = v_array[t0_model_index]   # km/s, vector

orbit_derived_inputs = (r_array, t_array, t0_model_index, lat_gp, lon_gp)


# 3) Read in the data files

timeRate = np.load(cwd + "/Data/RXTE/60079/7_9_matrices/7_9_rateTime.npy")
ampCenters = np.load(cwd + "/Data/RXTE/60079/7_9_matrices/7_9_ampCenters.npy")

time_data = timeRate[0, :]
rate_data = timeRate[1, :]
normalized_amplitudes = ampCenters[:, 0]
bin_centers = ampCenters[:, 1]

unattenuated_rate = 213  # 1852 for test_obs

eband_derived_inputs = (e_band, bin_size, normalized_amplitudes, bin_centers)
#
model_obj = TransmitModel(obs_dict, orbit_derived_inputs, eband_derived_inputs)
transmit_model, time_crossing_model = model_obj.calculate_transmit_model()

# 4)
model_and_data_tuple = (time_crossing_model, transmit_model, time_data, rate_data, unattenuated_rate)

comp_obj = CurveComparison(obs_dict, model_and_data_tuple)
t0_e, dt_e = comp_obj.t0_e, comp_obj.dt_e
del comp_obj