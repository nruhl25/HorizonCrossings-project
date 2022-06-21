# Author: Nathaniel Ruhl
# This script takes in an (RXTE) count rate array and returns the unattenuated count rate

import numpy as np


def get_unattenuated_rate_RXTE(obs_dict, rate_data, time_data, bin_size=1.0):
    # define a time range over which to get the average unattenuated rate
    if obs_dict['hc_type'] == 'setting':
        t_range = np.where((time_data >= obs_dict['crossing_time_range'][0]-50) & (time_data <= obs_dict['crossing_time_range'][0]+50))
    else:
        t_range = np.where((time_data >= obs_dict['crossing_time_range'][1]-50) & (time_data <= obs_dict['crossing_time_range'][1]+50))
    full_transmit = np.mean(rate_data[t_range])
    return full_transmit