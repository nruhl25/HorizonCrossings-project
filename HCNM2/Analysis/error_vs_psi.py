# Author: Nathaniel Ruhl
# This script looks at the effect of out-of-plane angle on the results of the RXTE horizon crossings

# Import standard libraries
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import Modules
from Modules.OrbitModel import OrbitModel
from Modules.LocateR0hc import LocateR0hc
from Modules.TransmitModel import TransmitModel
from Modules.generate_nans_rxte import generate_nans_rxte
from Preprocessing.RXTE_channel_to_keV import channel_to_keV_epoch5
from Modules.CurveComparison import CurveComparison
from Modules.get_unattenuated_rate_RXTE import get_unattenuated_rate_RXTE
from Modules.calculate_psi import calculate_psi

# Import observation dictionaries
from ObservationDictionaries.RXTE.all_dicts import *

cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd


# This function analyzes a single energy band of an RXTE horizon crossing
# OUTPUTS: t0_e, dt_e, t0_model
def RXTE_driver(obs_dict, e_band_ch):
    # 1) Define orbit model
    r_array, v_array, t_array = OrbitModel.define_orbit_model(obs_dict, "rossi", time_step=0.01)

    # 2) LocateR0hc (must know hc_type here, R_orbit and h_unit defined within the class)
    r0_obj = LocateR0hc(obs_dict, r_array, v_array, t_array)
    t0_model_index, lat_gp, lon_gp = r0_obj.return_r0_data()
    t0_model = t_array[t0_model_index]
    del r0_obj

    v0 = v_array[t0_model_index]  # km/s, vector

    orbit_derived_inputs = (r_array, t_array, t0_model_index, lat_gp, lon_gp)

    # 3a) Define energy band
    e_band_kev = channel_to_keV_epoch5(e_band_ch)
    bin_size = 1.0

    # vars below used for reading in the correct data files
    obsid = obs_dict["obsID"]
    e_id = f"{e_band_ch[0] + 1}_{e_band_ch[1]}"  # string identifier for the given energy band
    fn_rateTime = cwd + f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_rateTime.npy"
    fn_ampCenters = cwd + f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_ampCenters.npy"

    # 3b) Read in the data files
    rateTime = np.load(fn_rateTime)
    ampCenters = np.load(fn_ampCenters)

    rate_data_raw = rateTime[:, 0]
    time_data_raw = rateTime[:, 1]
    normalized_amplitudes = ampCenters[:, 0]
    bin_centers_kev = ampCenters[:, 1]  # keV

    unattenuated_rate = get_unattenuated_rate_RXTE(obs_dict, rate_data_raw, time_data_raw)
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
    return t0_e, dt_e, t0_model


def err_vs_psi(x, a, b):
    return a*x+b


def main():
    all_dicts = [dict_91802, dict_60079, dict_50099, dict_40805, dict_50098]
    obsID_list = []  # to be filled with obsid's
    Dt_list = []   # sec
    dt_list = []  # sec, standard error
    psi_list = []  # deg out of plane

    e_band_ch = [7-1, 9]
    e_band_kev = channel_to_keV_epoch5(e_band_ch)
    for obs_dict in all_dicts:
        t0_e, dt_e, t0_model = RXTE_driver(obs_dict, e_band_ch)
        psi = np.rad2deg(calculate_psi(obs_dict))

        plt.scatter(psi, abs(t0_e-t0_model), label=obs_dict['obsID'])

        # add all but 50098 to list
        if obs_dict['obsID'] != 50098:
            Dt_list.append(abs(t0_e-t0_model))
            dt_list.append(dt_e)
            psi_list.append(psi)
            obsID_list.append(obs_dict['obsID'])

    # Calculate and plot trendline
    popt, pcov = curve_fit(err_vs_psi, xdata=psi_list, ydata=Dt_list)
    a, b = popt
    x = np.linspace(min(psi_list), max(psi_list), 1000)
    y = err_vs_psi(x, a, b)
    plt.plot(x, y, label=fr"$\Delta t$ = {a:.2f}$\psi$+{b:.2f}")

    plt.title(f"RXTE Horizon Crossings of Crab Nebula ({e_band_kev[0]}-{e_band_kev[1]} keV)")
    plt.xlabel(r"Out-of-plane angle to source, $\psi$ (deg)")
    plt.ylabel(r"HCNM measurement error, $\Delta t$ (sec)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))