# Author: Nathaniel Ruhl
# This is the driver for RXTE analysis. We will create an example obs_dict

# Libraries used for the proof-of-concept with MSIS and the ellipsoid Earth
from Modules.LocateR0hc2 import LocateR0hc2
from Modules.TransmitModel2 import TransmitModel2
from Modules.CurveComparison import CurveComparison

# Libraries used for the Navigational Driver
from Modules.LocateR0hcNav import LocateR0hcNav
from Modules.FitAtmosphere import FitAtmosphere

# Aditional function to read and do some preliminary processing on RXTE data
from Modules.read_rxte_data import read_rxte_data


# import observation dictionary
from ObservationDictionaries.RXTE.all_dicts import all_dicts

import matplotlib.pyplot as plt
import numpy as np

def RXTE_MSIS_Driver(obs_dict, e_band_ch):
    #1) LocateR0hc2
    r02_obj = LocateR0hc2(obs_dict, "rossi")
    v0 = r02_obj.v0_model
    print(f"t0_model = {r02_obj.t0_model}")

    #2) Read and pre-process rxte data for obs_dict and e_band_ch
    bin_size=1.0
    rate_data, time_data, normalized_amplitudes, bin_centers_kev, unattenuated_rate, e_band_kev = read_rxte_data(obs_dict, e_band_ch)

    # 3) Calculate transmittance model
    eband_derived_inputs = (e_band_kev, bin_size,
                            normalized_amplitudes, bin_centers_kev)
    model_obj = TransmitModel2(r02_obj, eband_derived_inputs)
    transmit_model = model_obj.transmit_model
    time_crossing_model = model_obj.time_crossing_model   # Note that this is [0, time_final], not MET

    # 4) Curve Comparison
    model_and_data_tuple = (time_crossing_model, transmit_model,
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
    return 0

def RXTE_Nav_Driver(obs_dict, e_band_ch, h0_ref, orbit_model="rossi"):
    r0_obj = LocateR0hcNav(obs_dict, orbit_model, h0_ref)
    print(f"t0_model = {r0_obj.t0_model} sec")

    #2) Read in RXTE data for the specified energy band
    rate_data, time_data, normalized_amplitudes, bin_centers_kev, unattenuated_rate, e_band_kev = read_rxte_data(obs_dict, e_band_ch)

    #3) Fit count rate vs h (geocentric tangent altitudes above y0_ref)
    fit_obj = FitAtmosphere(obs_dict, orbit_model, r0_obj,
                rate_data, time_data, unattenuated_rate)

    print(f"c = {fit_obj.c_fit}")

    return fit_obj.c_fit

def main():
    # Choose observation
    obs_dict = all_dicts[0]
    e_band_ch = [7-1, 9]
    # RXTE_MSIS_Driver(obs_dict, e_band_ch)
    h50_list = []
    for dict in all_dicts:
        plt.scatter(1, RXTE_Nav_Driver(dict, e_band_ch, h0_ref=40), label=f'F10.7={dict["f107"]}')
    plt.plot(h50_list)
    plt.title("Channel 1")
    plt.ylabel(r"${h_{{50}}}$ (km) above reference sphere")
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    print(f"-------{(time.time() - start_time):.4f}sec")
