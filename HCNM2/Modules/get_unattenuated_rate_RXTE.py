# Author: Nathaniel Ruhl
# This script takes in an (RXTE) count rate array and returns the unattenuated count rate

import numpy as np
from scipy.optimize import curve_fit


def double_exponential(x, N, a, b):
    return N * np.exp(-np.exp(-a * x + b))


def get_unattenuated_rate_RXTE(rate_data, bin_size=1.0):
    time = np.arange(0, len(rate_data), bin_size)
    popt, pcov = curve_fit(double_exponential, time, rate_data)
    full_transmit = popt[0]

    # Identify the type of horizon crossing
    # if popt[1] > 0:
    #     hc_type_id = "rising"
    # elif popt[1] < 0:
    #     hc_type_id = "setting"

    return full_transmit