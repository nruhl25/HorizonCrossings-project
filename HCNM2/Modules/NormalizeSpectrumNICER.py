# Author: Nathaniel Ruhl
# This class reads NICER's .evt file and normalizes the source spectrum in the specified en_range for use in TransmitModel.py

import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

# INPUTS:
# evt_obj (object of ReadEVT, so we don't have to readd the same EVT file multiple times)
# e_band: [1.0, 2.0], energy with which to normaliz the spectrum

# IMPORTANT ATTRIBUTES:
# self.normalized_counts
# self.bin_centers  [keV]


class NormalizeSpectrumNICER:
    # class variables
    FREQ_MAX = 20  # max frequency index in the fourier transform of the smoothed spectrum
    DX_FULL_EBAND_INTEGRATION = 0.05  # keV, steps for calculating area under the user-inputted energy range
    DX_PROBABILITY_INTEGRATION = 0.0005  # keV, steps (within a step of constant cross section) to calc probability
    hist_bins = 100

    def __init__(self, evt_obj, e_band):
        # Unpack inputs
        self.obs_dict = evt_obj.obs_dict
        self.event_times = evt_obj.event_times
        self.event_energies_kev = evt_obj.event_energies_kev
        self.e_band = e_band
        self.en_lower_kev = self.e_band[0]    # [keV]
        self.en_upper_kev = self.e_band[1]    # [keV]

        # Split events into en_range and spectrum_time_range
        _, self.event_energies_spectrum_kev = self.reduce_spectrum_time_range()

        # Collect the spectrum corresponding to the spectrum time range
        self.amplitudes, self.bin_boundaries = np.histogram(self.event_energies_spectrum_kev, bins=NormalizeSpectrumNICER.hist_bins)
        self.bin_width = self.bin_boundaries[1] - self.bin_boundaries[0]  # width of energy bins in keV
        self.bin_centers = self.bin_boundaries[0:-1] + 0.5 * self.bin_width  # bin centers

        # Smooth the spectrum, normalize it in the desired range, and calculate the probability in the range
        self.amplitude_clean = self.smooth_spectrum()  # smooths self.amplitudes
        self.normalized_amplitudes = self.amplitude_clean / self.calc_total_area_eband()

    # Method called from the Driver: returns 2 fields
    def return_spectrum_data(self):
        return self.normalized_amplitudes, self.bin_centers

    # Used when binning the data by energy band within obs_dict['spectrum_time_range']
    def reduce_spectrum_time_range(self):
        time_range = self.obs_dict["spectrum_time_range"]
        start_index = np.where(self.event_times >= time_range[0])[0][0]
        stop_index = np.where(self.event_times >= time_range[1])[0][0]

        event_times_range = self.event_times[start_index:stop_index]
        event_energies_kev_range = self.event_energies_kev[start_index:stop_index]

        return event_times_range, event_energies_kev_range


    # method to smooth the spectrum with fft
    def smooth_spectrum(self):
        # Fourrier transform from the energy domain to the 1/energy domain
        N = len(self.bin_centers) - 1  # number of sampling points

        # Conduct Fourrier transform
        yf = fft(self.amplitudes)
        # Ignore the x-axis for now. More code in jupyter notebook

        yf_clean = yf

        yf_clean.real[NormalizeSpectrumNICER.FREQ_MAX:len(yf)] = 0
        yf_clean.imag[NormalizeSpectrumNICER.FREQ_MAX:len(yf)] = 0

        # Inverse fourrier transform
        amplitude_clean = ifft(yf_clean)

        return amplitude_clean.real

    # methods to normalize the spectrum. spectrum_counts is the y value
    def interpolate_spectrum(self, e_val, spectrum_counts):
        func = interp1d(self.bin_centers, spectrum_counts)
        return func(e_val)

    # calculates total area under the spectrum IN THE ENERGY RANGE
    def calc_total_area_eband(self):
        area = 0  # add up the total area
        dx = NormalizeSpectrumNICER.DX_FULL_EBAND_INTEGRATION

        for left_bound in np.arange(self.en_lower_kev, self.en_upper_kev, dx):
            # trapezoidal sum
            area += (self.interpolate_spectrum(left_bound, self.amplitude_clean) + self.interpolate_spectrum(left_bound + dx, self.amplitude_clean))*dx/2
        return area

    # FUNCTION: Takes in a range of energies in keV, returns a probability from the normalized spectrum
    # en1 and en2 are located inside self.en_lower_kev and self.en_upper_kev
    def calc_spectrum_probability(self, en1_kev, en2_kev):

        # en1 can't be below 0.27 keV because there are no counts below that
        if en1_kev < min(self.bin_boundaries):
            raise RuntimeError(f"en1 can't be smaller than {min(self.bin_boundaries)}")

        # perform numerical integration (trapezoidal Reimann sum) between en1 and en2
        dx = NormalizeSpectrumNICER.DX_PROBABILITY_INTEGRATION

        prob_in_range = 0  # this is the integral sum of probability. Add up area under the curve
        for left_bound in np.arange(en1_kev, en2_kev, dx):
            prob_in_range += (self.interpolate_spectrum(left_bound, self.normalized_amplitudes) + self.interpolate_spectrum(left_bound + dx, self.normalized_amplitudes))*(dx / 2)
        return prob_in_range

    # Code to plot the normalized spectrum
    def plot_normalized_spectrum(self):
        plt.plot(self.bin_centers, self.normalized_amplitudes, label=f'Spectrum normalized for {self.en_lower_kev} - {self.en_upper_kev} keV')
        
        plt.xlabel('Energy (keV)')
        plt.ylabel('Normalized Counts (Bin Size = %.2f keV)' % self.bin_width)
        plt.legend()
        plt.show()
        return 0

    # Numerical integration for the effective transmittance model
    @classmethod
    def set_dx_spectrum_probability(cls, dx):
        cls.DX_PROBABILITY_INTEGRATION = dx

    @classmethod
    def get_n_intervals_effective_transmit(cls):
        return cls.DX_PROBABILITY_INTEGRATION


# Example code to test this calss
if __name__ == '__main__':
    from ReadEVT import ReadEVT
    from ObservationDictionaries.NICER.crabNICER import crabNICER

    obs_dict = crabNICER
    e_band = [1.0, 2.0]  # keV

    evt_obj = ReadEVT(obs_dict)
    spec_obj = NormalizeSpectrumNICER(evt_obj, e_band)
    normalized_amplitudes = spec_obj.normalized_amplitudes
    bin_centers = spec_obj.bin_centers
    min_energy = spec_obj.bin_boundaries[0]

    import matplotlib.pyplot as plt
    spec_obj.plot_normalized_spectrum()
