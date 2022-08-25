# Author: Nathaniel Ruhl
# This script is used to determine the uncertainty in the tanh fit (dt50) vs count rate

# Libraries used for the Navigational Driver
from Modules.LocateR0hcNav import LocateR0hcNav
# curve comparison class is not necessary with the fit
from Modules.FitAtmosphere import FitAtmosphere

# Aditional function to read and do some preliminary processing on RXTE data
from Modules.read_rxte_data import read_rxte_data
from Modules import constants


# import observation dictionary
from ObservationDictionaries.RXTE.all_dicts import all_dicts

import matplotlib.pyplot as plt
import numpy as np

# This function cosiders the unce
def get_errors_RXTE(obs_dict, h0_ref, orbit_model="rossi"):
    # 1) Locate R0hc
    r0_obj = LocateR0hcNav(obs_dict, orbit_model, h0_ref)

    a1 = np.arange(6, 28, 3)
    a2 = np.arange(9, 31, 3)
    e_band_ch_array = np.column_stack((a1, a2))

    # Lists for dy50, dt50, and unattenuated_rate
    dy50_list = []
    dt50_list = []
    dt50_slide_list = []
    N0_list = []

    for e_band_ch in e_band_ch_array:

        #2) Read in RXTE data for the specified energy band
        rate_data, time_data, normalized_amplitudes, bin_centers_kev, unattenuated_rate, e_band_kev = read_rxte_data(
            obs_dict, e_band_ch)

        #3) Fit count rate vs h (geocentric tangent altitudes above y0_ref)
        fit_obj = FitAtmosphere(obs_dict, orbit_model, r0_obj,
                                rate_data, time_data, unattenuated_rate, e_band_kev)
        # fit_obj.plot_tanh_fit()
        N0_list.append(unattenuated_rate)
        dy50_list.append(fit_obj.dy50)
        dt50_list.append(fit_obj.dt50)
        dt50_slide_list.append(fit_obj.dt50_slide)
    return N0_list, dy50_list, dt50_list, dt50_slide_list


# Choose observation
# obs_dict = all_dicts[0]
plt.figure()
for obs_dict in all_dicts:
    N0_list, dy50_list, dt50_list, dt50_slide_list = get_errors_RXTE(obs_dict, h0_ref=40)
    plt.plot(N0_list, dt50_slide_list, '.', label=f'obsID={obs_dict["obsID"]}')

# create a trendline for the NICER data (AAS 22-073)
counts = np.linspace(300, 3500, 1000)
plt.plot(counts, 1.71/np.sqrt(counts), label='NICER results')
plt.legend()
plt.title("RXTE Crossing Measurement Uncertainties (all energy bands)")
plt.xlabel("Unattenuated Count Rate (ct/sec)")
plt.ylabel(r"Standard deviation of measurement, $\delta t_{e}$ (sec)")
plt.show()
