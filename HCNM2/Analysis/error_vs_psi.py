# Author: Nathaniel Ruhl
# This script looks at the effect of out-of-plane angle on the results of the RXTE horizon crossings

# Import standard libraries
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import Local Modules
from Modules.LocateR0hc2 import LocateR0hc2
from Modules.TransmitModel2 import TransmitModel2
from Modules.generate_nans_rxte import generate_nans_rxte
from Preprocessing.RXTE_channel_to_keV import channel_to_keV_epoch5
from Modules.CurveComparison import CurveComparison
from Modules.get_unattenuated_rate_RXTE import get_unattenuated_rate_RXTE

cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd

# Import observation dictionaries
from ObservationDictionaries.RXTE.all_dicts import all_dicts

# This function analyzes a single energy band of an RXTE horizon crossing
# OUTPUTS: t0_e, dt_e, t0_model
def RXTE_driver(obs_dict, e_band_ch):
    # 1) Define orbit model and locate r0_hc
    r02_obj = LocateR0hc2(obs_dict, "rossi")
    # r_array, v_array, t_array = OrbitModel.define_orbit_model(obs_dict, "rossi", time_step=0.01)

    # 2a) Define energy band
    e_band_kev = channel_to_keV_epoch5(e_band_ch)
    bin_size = 1.0

    # vars below used for reading in the correct data files
    obsid = obs_dict["obsID"]
    e_id = f"{e_band_ch[0] + 1}_{e_band_ch[1]}"  # string identifier for the given energy band
    fn_rateTime = cwd + f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_rateTime.npy"
    fn_ampCenters = cwd + f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_ampCenters.npy"

    # 2b) Read in the data files
    rateTime = np.load(fn_rateTime)
    ampCenters = np.load(fn_ampCenters)

    rate_data_raw = rateTime[:, 0]
    time_data_raw = rateTime[:, 1]
    normalized_amplitudes = ampCenters[:, 0]
    bin_centers_kev = ampCenters[:, 1]  # keV

    unattenuated_rate = get_unattenuated_rate_RXTE(obs_dict, rate_data_raw, time_data_raw)
    print(f"unattenuated rate = {unattenuated_rate}")

    # 3) Lengthen rate_data and time_data if necessary
    rate_data, time_data = generate_nans_rxte(obs_dict, rate_data_raw, time_data_raw)

    eband_derived_inputs = (e_band_kev, bin_size,
                            normalized_amplitudes, bin_centers_kev)

    # 4) Calculate transmittance model
    model2_obj = TransmitModel2(r02_obj, eband_derived_inputs)
    transmit_model = model2_obj.transmit_model
    # Note that this is [0, time_final], not MET
    time_crossing_model = model2_obj.time_crossing_model

    # 5) Curve Comparison
    model_and_data_tuple = (time_crossing_model, transmit_model, time_data, rate_data, unattenuated_rate)
    comp_obj = CurveComparison(obs_dict, model_and_data_tuple)
    t0_e, dt_e = comp_obj.t0_e, comp_obj.dt_e
    del comp_obj
    return t0_e, dt_e, r02_obj.t0_model, r02_obj.psi_deg


def err_vs_psi(x, a, b):
    return a*x+b


def main():
    # Create array with energy band channels for RXTE
    a1 = np.arange(6, 28, 3)
    a2 = np.arange(9, 31, 3)
    e_band_ch_array = np.column_stack((a1, a2))

    for e_band_ch in e_band_ch_array:
        e_band_kev = channel_to_keV_epoch5(e_band_ch)
        plt.figure()
    
        obsID_list = []  # to be filled with obsid's
        Dt_list = []   # sec
        dt_list = []  # sec, standard error
        psi_list = []  # deg out of plane

        for obs_dict in all_dicts:
            t0_e, dt_e, t0_model, psi_deg = RXTE_driver(obs_dict, e_band_ch)

            plt.scatter(psi_deg, abs(t0_e-t0_model), label=obs_dict['obsID'])

            # Don't use 50098 in the trendline
            if obs_dict['obsID'] != 50098:
                Dt_list.append(abs(t0_e-t0_model))
                dt_list.append(dt_e)
                psi_list.append(psi_deg)
                obsID_list.append(obs_dict['obsID'])

        # Calculate and plot trendline
        popt, pcov = curve_fit(err_vs_psi, xdata=psi_list, ydata=Dt_list)
        a, b = popt
        x = np.linspace(min(psi_list), max(psi_list), 1000)
        y = err_vs_psi(x, a, b)
        plt.plot(x, y, label=fr"$\Delta t$ = {a:.3f}$\psi$+{b:.3f}")

        plt.title(f"RXTE Horizon Crossings of Crab Nebula ({e_band_kev[0]}-{e_band_kev[1]} keV)")
        plt.xlabel(r"Out-of-plane angle to source, $\psi$ (deg)")
        plt.ylabel(r"HCNM measurement error, $\Delta t$ (sec)")
        plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
