# This is the first script to run when analyzing a new NICER horizon crossing.
# It is used in order to define obs_dict["time_range_crossing"], a ~500 sec time range around the crossing

# import local modules
from Modules import tools as tools

# import standard libraries
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt


class ReadEVT:

    def __init__(self, evt_path):
        self.evt_path = evt_path
        self.event_times, self.event_energies_kev = self.readEVT()  # (full time range) used to create spectra

    # This methods reads the evt file and returns 2 arrays: events and times
    def readEVT(self):
        tab_evt = Table.read(self.evt_path, hdu=1)
        event_times = np.array(tab_evt["TIME"])
        event_energies_kev = np.array(tab_evt['PI']) / 100
        return event_times, event_energies_kev

    # Method to call to bin data within a single energy band
    def bin_data_eband(self, e_band, bin_size=1):
        e_band_bool_array = ((self.event_energies_kev >= e_band[0]) &
                             (self.event_energies_kev < e_band[1]))

        band_count_rate, band_times_binned = ReadEVT.calc_count_rate(self.event_times, e_band_bool_array,
                                                                     bin_size)
        return band_count_rate, band_times_binned

    @staticmethod
    def calc_count_rate(time_array, e_band_bool_array, bin_size):
        # Note that the two arrays must correspond to the same time interval
        binCounts = []
        binTime = []
        for time_bin in np.arange(min(time_array), max(time_array)+bin_size, bin_size):
            desind = np.where((time_array >= time_bin) & (time_array < time_bin + bin_size))
            binCounts.append(np.sum(e_band_bool_array[desind[0]]))
            binTime.append(time_bin+(bin_size/2))
        return np.array(binCounts), np.array(binTime)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    # Note that we haven't yet created the dictionary for the observation at this point...
    root_path = '/Users/nathanielruhl/Documents/HorizonCrossings-Summer22/HCNM2'
    event_obj = ReadEVT(root_path + '/Data/NICER/2-3-20-v4641/NICER_events.evt')

    e_band_array = np.array([[1.0, 2.0],
                             [3.0, 5.0]])
    plt.figure()  # Identify the ~500 second time range in this plot

    for e_band in e_band_array:
        band_count_rate, binned_times = event_obj.bin_data_eband(e_band)
        plt.plot(binned_times, band_count_rate, label=f"{e_band[0]} - {e_band[1]} keV")
    plt.legend()
    plt.show()

    # get the datetime of the desired crossing time range
    print(tools.convert_time_NICER(480 + 2.40148e8))
