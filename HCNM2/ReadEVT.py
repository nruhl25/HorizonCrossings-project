# Author: Nathaniel Ruhl
# This script includes a class to read NICER'S .evt file. The order in which things are defined and used is

from astropy.table import Table
import numpy as np
from scipy.optimize import curve_fit


# This class reads the entire .evt file, and contains a method to split the events by e_band before binning
# First instantiated before creating the normalized spectrum
class ReadEVT:

    def __init__(self, obs_dict):
        self.obs_dict = obs_dict
        self.event_times, self.event_energies_kev = self.readEVT()  # (full time range) used to create spectra

    # This methods reads the evt file and returns 2 arrays: events and times
    def readEVT(self):
        tab_evt = Table.read(self.obs_dict["evt_path"], hdu=1)
        event_times = np.array(tab_evt["TIME"])
        event_energies_kev = np.array(tab_evt['PI']) / 100
        return event_times, event_energies_kev

    # Method called to bin the data within e_band and in a given time range
    # itime_range_crossing: True uses time_range_500 (in obs_dict)
    def return_crossing_data(self, e_band, bin_size, t500_toggle=True):

        band_event_times, e_band_events_kev, e_band_bool_array = self.en_split(e_band, t500_toggle)

        band_count_rate, band_times_binned = \
                ReadEVT.calc_count_rate(band_event_times, e_band_bool_array, bin_size)

        unnatenuated_rate, hc_type_id = self.get_band_max_count_rate(band_count_rate, bin_size)

        '''
        # This method lengthens the data arrays so they contain t0 if looking at the crossing time range
        if t500_toggle == True:
            band_count_rate, band_times_binned = self.generate_nans(band_count_rate, band_times_binned, bin_size)
        '''
        return band_count_rate, band_times_binned, unnatenuated_rate

    #### The methods below are helper functions for self.return_crossing_data() ####

    # Used when binning the data by energy band within obs_dict['crossing_time_range']
    # t500_toggle=False: bin the entire data file
    def en_split(self, e_band, t500_toggle=True):
        if t500_toggle is True:
            time_range_500 = self.obs_dict["crossing_time_range"]
            start_index_500 = np.where(self.event_times >= time_range_500[0])[0][0]
            stop_index_500 = np.where(self.event_times >= time_range_500[1])[0][0]

            event_times_range = self.event_times[start_index_500:stop_index_500]
            event_energies_kev_range = self.event_energies_kev[start_index_500:stop_index_500]
        else:
            event_times_range = self.event_times
            event_energies_kev_range = self.event_energies_kev

        bool_array = ((event_energies_kev_range >= e_band[0]) &
                      (event_energies_kev_range < e_band[1]))
        indices = np.where((event_energies_kev_range >= e_band[0]) & (event_energies_kev_range < e_band[1]))

        return event_times_range[indices[0]], event_energies_kev_range[indices[0]], bool_array

    @staticmethod
    def calc_count_rate(time_array_no_band, e_band_bool_array, bin_size):
        # Note that the two arrays must correspond to the same time interval
        binCounts = []
        binTime = []
        for time_bin in np.arange(min(time_array_no_band), max(time_array_no_band)+bin_size, bin_size):
            desind = np.where((time_array_no_band >= time_bin) & (time_array_no_band < time_bin + bin_size))
            binCounts.append(np.sum(e_band_bool_array[desind[0]]))
            binTime.append(time_bin+(bin_size/2))
        return np.array(binCounts), np.array(binTime)

    # This function is used to lengthen the data array if t0 is not in the evt file
    # (not currently used because it's not working correctly)
    def generate_nans(self, band_count_rate, band_times_binned, bin_size):
        start_crossing = self.obs_dict["crossing_time_range"][0]
        stop_crossing = self.obs_dict["crossing_time_range"][1]
        hc_type = self.obs_dict["hc_type"]

        # If we identified a start_crossing or stop_crossing point that is not in the data file
        if hc_type == "rising":
            if start_crossing < band_times_binned[0]:
                delta_T = int(np.ceil(band_times_binned[0] - start_crossing))
                zero_count_rates = np.empty(int(delta_T/bin_size))
                zero_count_rates[:] = np.nan
                times_before_start = np.arange(band_times_binned[0] - delta_T, band_times_binned[0], bin_size)

                # insert in front of the list
                band_count_rate = np.insert(band_count_rate, 0, zero_count_rates)
                band_times_binned = np.insert(band_times_binned, 0, times_before_start)
            else:
                pass
        elif hc_type == "setting":
            if stop_crossing > band_times_binned[-1]:
                delta_T = int(np.ceil(band_times_binned[-1] - stop_crossing))
                zero_count_rates = np.empty(int(delta_T/bin_size))
                zero_count_rates[:] = np.nan
                times_before_start = np.arange(band_times_binned[-1] - delta_T, band_times_binned[-1], bin_size)

                # insert at end of the list
                band_count_rate = np.append(band_count_rate, zero_count_rates)
                band_times_binned = np.append(band_times_binned, times_before_start)
            else:
                pass

        return band_count_rate, band_times_binned

    #### Functions to identify the unnatenuated count rate

    def double_exponential(self, x, N, a, b):
        return N*np.exp(-np.exp(-a * x + b))

    def get_band_max_count_rate(self, band_count_rate, bin_size):
        rate_to_fit = np.nan_to_num(band_count_rate, nan=0)
        # Only works after you have already shortened rate_to_fit into the approximate HC time range
        max_rate_index = np.where(rate_to_fit == np.max(rate_to_fit))[0][0]
        # calculate the mean value for a time range around the max value.
        # Indices cuts out the noise and zero count rate at the end
        indices = np.where(rate_to_fit[max_rate_index:len(rate_to_fit)] > np.max(rate_to_fit)/2)[0]
        plateau_rate = np.mean(rate_to_fit[max_rate_index+indices])

        # Improve plateau rate -- go 40 seconds before and after 50% point
        # This step is why generate_nans() is used. Index1 could be negative if the data doesn't go far back
        index1 = np.where(rate_to_fit/plateau_rate >= 0.5)[0][0] - int(40/bin_size)
        if index1 < 0:
            raise RuntimeError('The data starts less than 40/bin_size seconds before the 50% point')
        index2 = np.where(rate_to_fit/plateau_rate >= 0.5)[0][0] + int(40/bin_size)
        rate_to_fit = rate_to_fit[index1:index2]
        time_to_fit = np.linspace(0, 200, len(rate_to_fit))
        popt, pcov = curve_fit(self.double_exponential, time_to_fit, rate_to_fit)
        full_transmit = popt[0]

        # Identify the type of horizon crossing
        if popt[1] > 0:
            hc_type_id = "rising"
        elif popt[1] < 0:
            hc_hc_type_id = "setting"

        return full_transmit, hc_type_id


if __name__ == '__main__':
    from ObservationDictionaries.v4641 import v4641 as obs_dict
    import matplotlib.pyplot as plt

    obj = ReadEVT(obs_dict)
    band_count_rate, band_times_binned, max_rate = obj.bin_eband_data([1.0, 2.0], 1, t500_toggle=True)
    print(max_rate)
    plt.plot(band_times_binned, band_count_rate, ".")
    plt.show()
