# This script contains functions to convert from channel to keV for the RXTE data

import numpy as np
from scipy.interpolate import interp1d


def channel_to_keV_epoch5(channel_in):
    en_channel = np.arange(7.0, 40.0, 1.0)

    en_kev = [
        3.28, 3.68, 4.09, 4.49, 4.90, 5.31, 5.71,
        6.12, 6.53, 6.94, 7.35, 7.76, 8.17, 8.57,
        8.98, 9.40, 9.81, 10.22, 10.63, 11.04, 11.45,
        11.87, 12.28, 12.69, 13.11, 13.52, 13.93,
        14.35, 14.76, 15.18, 15.60, 16.01, 16.43
    ]

    ch_to_kev = interp1d(x=en_channel, y=np.array(en_kev), kind="linear", fill_value="extrapolate")

    return ch_to_kev(np.array(channel_in, float))
