# Author: Nathaniel Ruhl
# This function is used to lengthen the rate_data and time_data arrays by generating nans (similar to readEVT.py)
# The curve comparison in CurveComparison.py requires that t0 is in self.time_data

import numpy as np


# This function is used to lengthen the data array if t0 is not in the evt file
def generate_nans_rxte(obs_dict, rate_data, time_data, bin_size=1):
    start_crossing = obs_dict["crossing_time_range"][0]
    stop_crossing = obs_dict["crossing_time_range"][1]
    hc_type = obs_dict["hc_type"]

    # If we identified a start_crossing or stop_crossing point that is not in the data file
    rate_data = np.array(rate_data, float)  # integers can't go with np.nan
    if hc_type == "rising":
        if start_crossing < time_data[0]:
            delta_T = int(np.ceil(time_data[0] - start_crossing))
            zero_count_rates = np.full(int(delta_T/bin_size), np.nan)
            times_before_start = np.arange(time_data[0] - delta_T, time_data[0], bin_size)

            # insert in front of the list
            rate_data = np.insert(rate_data, 0, zero_count_rates, axis=0)
            time_data = np.insert(time_data, 0, times_before_start, axis=0)
        else:
            pass
    elif hc_type == "setting":
        if stop_crossing > time_data[-1]:
            delta_T = int(np.ceil(stop_crossing - time_data[-1]))
            zero_count_rates = np.full(int(delta_T/bin_size), np.nan)
            times_before_start = np.arange(time_data[-1], time_data[-1] + delta_T, bin_size)

            # insert at end of the list
            rate_data = np.append(rate_data, zero_count_rates, axis=0)
            time_data = np.append(time_data, times_before_start, axis=0)
        else:
            pass

    return rate_data, time_data