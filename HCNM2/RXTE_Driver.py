# Author: Nathaniel Ruhl
# This is the driver for RXTE analysis. We will create an example obs_dict

from Modules.OrbitModel import OrbitModel
from Modules.LocateR0hc import LocateR0hc
from Modules.TransmitModel import TransmitModel
from Modules.generate_nans_rxte import generate_nans_rxte
from Preprocessing.RXTE_channel_to_keV import channel_to_keV_epoch5
from Modules.CurveComparison import CurveComparison
from Modules.get_unattenuated_rate_RXTE import get_unattenuated_rate_RXTE

# import observation dictionary
from ObservationDictionaries.RXTE.dict_60079 import dict_60079

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[0])  # HCNM2/ is cwd

# Choose observation
obs_dict = dict_60079

# 1) Define orbit model
r_array, v_array, t_array = OrbitModel.define_orbit_model(obs_dict, "rossi", time_step=0.01)

# 2) LocateR0hc (must know hc_type here, R_orbit and h_unit defined within the class)
r0_obj = LocateR0hc(obs_dict, r_array, v_array, t_array)
t0_model_index, lat_gp, lon_gp = r0_obj.return_r0_data()
t0_model = t_array[t0_model_index]
del r0_obj

v0 = v_array[t0_model_index]   # km/s, vector

orbit_derived_inputs = (r_array, t_array, t0_model_index, lat_gp, lon_gp)

# 3a) Choose energy band to analyze

e_band_ch = [7-1, 9]   # min to max energy of data included (minus one is important in lower bound)
e_band_kev = channel_to_keV_epoch5(e_band_ch)
print(f"e_band_kev = {e_band_kev}")
bin_size = 1.0

# vars below used for reading in the correct data files
obsid = obs_dict["obsID"]
e_id = f"{e_band_ch[0]+1}_{e_band_ch[1]}"  # string identifier for the given energy band
fn_rateTime = cwd + f"/Data/RXTE/{obsid}/{e_id}_matrices/{e_id}_rateTime.npy"
fn_ampCenters = cwd + f"/Data/RXTE/{obsid}/{e_id}_matrices/{e_id}_ampCenters.npy"


# 3b) Read in the data files
rateTime = np.load(fn_rateTime)
ampCenters = np.load(fn_ampCenters)

rate_data_raw = rateTime[:, 0]
time_data_raw = rateTime[:, 1]
normalized_amplitudes = ampCenters[:, 0]
bin_centers_ch = ampCenters[:, 1]   # channels
bin_centers_kev = channel_to_keV_epoch5(bin_centers_ch)

unattenuated_rate = 2626  # get_unattenuated_rate_RXTE(rate_data_raw) (this is under-estimating?)
print(f"unattenuated rate = {unattenuated_rate}")

# 4) Lengthen rate_data and time_data if necessary
rate_data, time_data = generate_nans_rxte(obs_dict, rate_data_raw, time_data_raw)

# 5) Calculate transmittance model
eband_derived_inputs = (e_band_kev, bin_size, normalized_amplitudes, bin_centers_kev)
model_obj = TransmitModel(obs_dict, orbit_derived_inputs, eband_derived_inputs)
transmit_model, time_crossing_model = model_obj.calculate_transmit_model()

# 6) Curve Comparison
model_and_data_tuple = (time_crossing_model, transmit_model, time_data, rate_data, unattenuated_rate)
comp_obj = CurveComparison(obs_dict, model_and_data_tuple)
t0_e, dt_e = comp_obj.t0_e, comp_obj.dt_e
del comp_obj
print(f"Geometrically, t_0 = {t_array[t0_model_index]:.2f}")

import matplotlib.pyplot as plt
plt.title("Horizon Crossing of Crab Nebula (RXTE)")
plt.ylabel("counts/sec")
plt.xlabel("Seconds (MET)")
plt.plot(time_data, rate_data, ".", label=f"{e_band_kev[0]:.2f}-{e_band_kev[1]:.2f} keV")
plt.plot(t0_e + time_crossing_model - time_crossing_model[-1], unattenuated_rate*transmit_model, label=fr"$t_{{0,e}}$ = {t0_e:.2f} +/- {dt_e:.2f}")
plt.legend()
plt.show()