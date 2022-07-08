# Author: Nathaniel Ruhl
# This is the driver for RXTE analysis. We will create an example obs_dict

import matplotlib.pyplot as plt
from Modules.OrbitModel import OrbitModel
from Modules.LocateR0hc import LocateR0hc
from Modules.LocateR0hc2 import LocateR0hc2
from Modules.TransmitModel import TransmitModel
from Modules.TransmitModel2 import TransmitModel2
from Modules.CurveComparison import CurveComparison
from Modules.generate_nans_rxte import generate_nans_rxte
from Preprocessing.RXTE_channel_to_keV import channel_to_keV_epoch5
from Modules.get_unattenuated_rate_RXTE import get_unattenuated_rate_RXTE

# import observation dictionary
from ObservationDictionaries.RXTE.dict_60079 import dict_60079
from ObservationDictionaries.RXTE.dict_50098 import dict_50098
from ObservationDictionaries.RXTE.dict_40805 import dict_40805
from ObservationDictionaries.RXTE.dict_91802 import dict_91802
from ObservationDictionaries.RXTE.dict_50099 import dict_50099

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[0])  # HCNM2/ is cwd

# Choose observation
obs_dict = dict_50099 # dict_60079

# 1) Define orbit model
r_array, v_array, t_array = OrbitModel.define_orbit_model(
    obs_dict, "rossi", time_step=0.01)

# 2) LocateR0hc (must know hc_type here, R_orbit and h_unit defined within the class)
r0_obj = LocateR0hc(obs_dict, r_array, v_array, t_array)
t0_model_index, lat_gp, lon_gp = r0_obj.return_r0_data()
t0_model = t_array[t0_model_index]
del r0_obj

r02_obj = LocateR0hc2(obs_dict, "rossi")
print(f"t0_model2 = {r02_obj.t0_model}")
print(f"t0_model1 = {t_array[t0_model_index]:.2f}")

v0 = v_array[t0_model_index]   # km/s, vector

orbit_derived_inputs = (r_array, t_array, t0_model_index, lat_gp, lon_gp)

# 3a) Choose energy band to analyze

# min to max energy of data included (minus one is important in lower bound)
e_band_ch = [7-1, 9]
e_band_kev = channel_to_keV_epoch5(e_band_ch)
print(f"e_band_kev = {e_band_kev}")
bin_size = 1.0

# vars below used for reading in the correct data files
obsid = obs_dict["obsID"]
# string identifier for the given energy band
e_id = f"{e_band_ch[0]+1}_{e_band_ch[1]}"
fn_rateTime = cwd + \
    f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_rateTime.npy"
fn_ampCenters = cwd + \
    f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_ampCenters.npy"


# 3b) Read in the data files
rateTime = np.load(fn_rateTime)
ampCenters = np.load(fn_ampCenters)

rate_data_raw = rateTime[:, 0]
time_data_raw = rateTime[:, 1]
normalized_amplitudes = ampCenters[:, 0]
bin_centers_kev = ampCenters[:, 1]   # keV

unattenuated_rate = get_unattenuated_rate_RXTE(
    obs_dict, rate_data_raw, time_data_raw)
# print(f"unattenuated rate = {unattenuated_rate}")

# 4) Lengthen rate_data and time_data if necessary
rate_data, time_data = generate_nans_rxte(
    obs_dict, rate_data_raw, time_data_raw)

# 5) Calculate transmittance model
eband_derived_inputs = (e_band_kev, bin_size,
                        normalized_amplitudes, bin_centers_kev)
model_obj = TransmitModel(obs_dict, orbit_derived_inputs, eband_derived_inputs)
transmit_model, time_crossing_model = model_obj.calculate_transmit_model()
model2_obj = TransmitModel2(r02_obj, eband_derived_inputs)
transmit_model2 = model2_obj.transmit_model
time_crossing_model2 = model2_obj.time_crossing_model   # Note that this is [0, time_final], not MET

# 6) Curve Comparison
model_and_data_tuple = (time_crossing_model2, transmit_model2,
                        time_data, rate_data, unattenuated_rate)
comp_obj = CurveComparison(obs_dict, model_and_data_tuple)
t0_e, dt_e = comp_obj.t0_e, comp_obj.dt_e
del comp_obj

plt.title("Horizon Crossing of Crab Nebula (RXTE)")
plt.ylabel("counts/sec")
plt.xlabel("Seconds (MET)")
plt.plot(time_data, rate_data, ".",
         label=f"{e_band_kev[0]:.2f}-{e_band_kev[1]:.2f} keV")
plt.plot(t0_e + time_crossing_model -
         time_crossing_model[-1], unattenuated_rate*transmit_model, label=fr"$t_{{0,e}}$ = {t0_e:.2f} +/- {dt_e:.2f}")
plt.legend()
plt.show()
