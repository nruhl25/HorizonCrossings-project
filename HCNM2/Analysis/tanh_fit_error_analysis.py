# Author: Nathaniel Ruhl
# This script is used to determine the uncertainty in the tanh fit (dt50) vs count rate

# Libraries used for the Navigational Driver
from Modules.LocateR0hcNav import LocateR0hcNav
# curve comparison class is not necessary with the fit
from Modules.FitAtmosphere import FitAtmosphere

# Specific for NICER analysis
from Modules.ReadEVT import ReadEVT
from Modules.NormalizeSpectrumNICER import NormalizeSpectrumNICER

# Aditional function to read and do some preliminary processing on RXTE data
from Modules.read_rxte_data import read_rxte_data
from Modules import constants


# import observation dictionary
from ObservationDictionaries.RXTE.all_dicts import all_dicts
from ObservationDictionaries.NICER.v4641NICER import v4641NICER
from ObservationDictionaries.NICER.crabNICER import crabNICER
NICER_dicts = [v4641NICER, crabNICER]

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


def get_errors_NICER(obs_dict, h0_ref, orbit_model='mkf'):
    r0_obj = LocateR0hcNav(obs_dict, orbit_model, h0_ref)

    # Lists for dy50, dt50, and unattenuated_rate
    dy50_list = []
    dt50_list = []
    dt50_slide_list = []
    N0_list = []

    bin_size = 1.0
    e_band_array = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    #2) Bin the NICER data
    evt_obj = ReadEVT(obs_dict)
    for e_band_kev in e_band_array:
        rate_data, time_data, unattenuated_rate = evt_obj.return_crossing_data(
            e_band_kev, bin_size)

        spec_obj = NormalizeSpectrumNICER(evt_obj, e_band_kev)
        normalized_amplitudes, bin_centers = spec_obj.return_spectrum_data()
        del spec_obj

        eband_derived_inputs = (
            e_band_kev, bin_size, normalized_amplitudes, bin_centers)

        #3) Fit count rate vs h (geocentric tangent altitudes above y0_ref)
        fit_obj = FitAtmosphere(obs_dict, orbit_model, r0_obj,
                                rate_data, time_data, unattenuated_rate, e_band_kev)
        N0_list.append(unattenuated_rate)
        dy50_list.append(fit_obj.dy50)
        dt50_list.append(fit_obj.dt50)
        dt50_slide_list.append(fit_obj.dt50_slide)
    return N0_list, dy50_list, dt50_list, dt50_slide_list


plt.figure()
for obs_dict in all_dicts:
    N0_list, dy50_list, dt50_list, dt50_slide_list = get_errors_RXTE(obs_dict, h0_ref=40)
    plt.plot(N0_list, dt50_slide_list, '.', label=f'obsID={obs_dict["obsID"]}')

# Add NICER data to the plot
# for obs_dict in NICER_dicts[1:]:
#     # tanh() fit is sensitive to low count rates. The v4641 Sgr crossing fit doesn't work
#     N0_list, dy50_list, dt50_list, dt50_slide_list = get_errors_NICER(
#         obs_dict, h0_ref=40)
#     plt.plot(N0_list, dt50_slide_list, '.', label='NICER Crab')

# create a trendline for the NICER data (AAS 22-073)
# counts = np.linspace(20, 5400, 1000)
# plt.plot(counts, 1.71/np.sqrt(counts), label='NICER results')
plt.legend()
# plt.title("RXTE Crossing Measurement Uncertainties (all energy bands)")
plt.xlabel("Unattenuated count rate (cts/sec)")
plt.ylabel(r"Measurement uncertainty, $\delta t_{e}$ (sec)")
plt.show()
