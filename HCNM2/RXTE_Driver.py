# Author: Nathaniel Ruhl
# This is the driver for RXTE analysis. We will create an example obs_dict

# Libraries used for the proof-of-concept with MSIS and the ellipsoid Earth
from Modules.LocateR0hc2 import LocateR0hc2
from Modules.TransmitModel2 import TransmitModel2
from Modules.CurveComparison import CurveComparison

# Libraries used for the Navigational Driver
from Modules.LocateR0hcNav import LocateR0hcNav
from Modules.FitAtmosphere import FitAtmosphere  # curve comparison class is not necessary with the fit

# Aditional function to read and do some preliminary processing on RXTE data
from Modules.read_rxte_data import read_rxte_data
from Modules import constants


# import observation dictionary
from ObservationDictionaries.RXTE.all_dicts import all_dicts

import matplotlib.pyplot as plt
import numpy as np

def RXTE_MSIS_Driver(obs_dict, e_band_ch, plot_color='tab:blue'):
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

    # plt.title("Horizon Crossing of Crab Nebula (RXTE)")
    plt.ylabel("Count Rate (counts/sec)")
    plt.xlabel("Seconds (MET)")
    plt.plot(time_data, rate_data, ".",
            label=f"{e_band_kev[0]:.2f}-{e_band_kev[1]:.2f} keV", color=plot_color)
    plt.plot(t0_e + time_crossing_model -
             time_crossing_model[-1], unattenuated_rate*transmit_model, color=plot_color) # label = fr"$t_{{0,e}}$ = {t0_e:.2f} +/- {dt_e:.2f}"
    plt.legend()

    print(f"t0_model={r02_obj.t0_model}")
    print(f"t0_e={t0_e}+/-{dt_e}")
    print(f"Delta t0 = {r02_obj.t0_model-t0_e}")
    print(f"r0={r02_obj.r0_hc}")
    return 0

# h50_expected is filter estimate

def RXTE_Nav_Driver(obs_dict, e_band_ch, h0_ref, orbit_model="rossi"):
    r0_obj=LocateR0hcNav(obs_dict, orbit_model, h0_ref)

    #2) Read in RXTE data for the specified energy band
    rate_data, time_data, normalized_amplitudes, bin_centers_kev, unattenuated_rate, e_band_kev=read_rxte_data(obs_dict, e_band_ch)
    transmit_data = rate_data/unattenuated_rate

    #3) Fit count rate vs h (geocentric tangent altitudes above y0_ref)
    fit_obj=FitAtmosphere(obs_dict, orbit_model, r0_obj, rate_data, time_data, unattenuated_rate, e_band_kev)

    # fit_obj.plot_tanh_fit()  # if you want to see the fit

    #4) Determine r50 and t50_model by changing h0_ref in LocateR0hcNav
    r50_obj = LocateR0hcNav(obs_dict, orbit_model, h0_ref=fit_obj.h50_fit+h0_ref)
    r50_model = r50_obj.r0_hc
    t50_model = r50_obj.t0_model

    print(f"Channel {e_band_ch}: ")
    print(f"h50_adm = {fit_obj.h50_predicted}")
    print(f"h50_measured = {fit_obj.h50_fit} +/- {fit_obj.dy50} km")
    print(f"t50_from_h50={fit_obj.t50_fit} +/- {fit_obj.dt50} sec")
    print(f"t50_newton = {fit_obj.t50_newton} sec")
    print(f"t50_model = {t50_model} sec")
    print(f"Delta t50 = {t50_model - fit_obj.t50_fit} sec")
    dt50_slide = fit_obj.get_dt50_slide()
    print(f"dt50_slide={dt50_slide} sec")
    return 0

def main():
    # Choose observation
    obs_dict = all_dicts[1]
    e_band_ch = [7-1, 9]
    print(f"Navigation Driver: ")
    # Big picture: h0_ref is really only used to get a preliminary idea of where the crossing is in the orbit
    RXTE_Nav_Driver(obs_dict, e_band_ch, h0_ref=40)

    print(f"MSIS_Driver: ")
    RXTE_MSIS_Driver(obs_dict, e_band_ch)

    # Plot showing multiple energy bands with colors
    # a1 = np.arange(6, 28, 3)
    # a2 = np.arange(9, 31, 3)
    # e_band_ch_array = np.column_stack((a1, a2))
    # colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    # i=0
    # for e_band_ch in e_band_ch_array:
    #     RXTE_MSIS_Driver(obs_dict, e_band_ch, plot_color=colors[i])
    #     i+=1
    # plt.show()
    return 0

if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    print(f"-------{(time.time() - start_time):.4f}sec")
