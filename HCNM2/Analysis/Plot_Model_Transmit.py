# Author: Nathaniel Ruhl
# This script plots the model transmittance curve as a function of time

import matplotlib.pyplot as plt
import numpy as np

# import local modules for HCNM Analysis
from Modules.OrbitModel import OrbitModel
from Modules.LocateR0hc import LocateR0hc
from Modules.ReadEVT import ReadEVT
from Modules.NormalizeSpectrumNICER import NormalizeSpectrumNICER
from Modules.TransmitModel import TransmitModel

# import observation dictionaries
from ObservationDictionaries.NICER.v4641NICER import v4641NICER

obs_dict = v4641NICER
bin_size = 1.0   # sec
e_band_array = np.array([[1.0, 2.0],
                         [2.0, 3.0],
                         [3.0, 4.0],
                         [4.0, 5.0]])

# 1) Define orbit model
r_array, v_array, t_array = OrbitModel.define_orbit_model(obs_dict, "mkf", time_step=0.01)

# 2) LocateR0hc (must know hc_type here (from obs_dict or step 1))
r0_obj = LocateR0hc(obs_dict, r_array, v_array, t_array)
t0_model_index, lat_gp, lon_gp = r0_obj.return_r0_data()
del r0_obj

orbit_derived_inputs = (r_array, t_array, t0_model_index, lat_gp, lon_gp)

for e_band in e_band_array:
    evt_obj = ReadEVT(obs_dict)
    rate_data, time_data, unattenuated_rate = evt_obj.return_crossing_data(e_band, bin_size)

    spec_obj = NormalizeSpectrumNICER(evt_obj, e_band)
    normalized_amplitudes, bin_centers = spec_obj.return_spectrum_data()
    del evt_obj
    del spec_obj

    eband_derived_inputs = (e_band, bin_size, normalized_amplitudes, bin_centers)

    # 5) RXTE Analysis starts here.
    model_obj = TransmitModel(obs_dict, orbit_derived_inputs, eband_derived_inputs)
    transmit_model, time_crossing_model = model_obj.calculate_transmit_model()

    plt.plot(time_crossing_model, transmit_model, label=f"{e_band[0]}-{e_band[1]} keV")

plt.legend()
plt.title("Transmittance vs. Time")
plt.show()