# Author: Nathaniel Ruhl
# This script contains a class to calculate the model transmission curve based on
# obs_dict and the specified orbit model

import numpy as np
from scipy.interpolate import interp1d

from Modules.xsects import BCM  # X-ray cross-sections
import Modules.tools as tools
import Modules.MSIS as MSIS


# MAIN FUNCTION: TransmitModel.calculate_transmit_model()

# INPUTS (9 total):
# obs_dict
# orbit_derived_inputs = (r_array, t_array, t0_model_index, lat_gp, lon_gp)
# eband_derived_inputs = (e_band, bin_size, normalized_amplitudes, bin_centers).
# normalized_amplitudes and bin_centers: Information from source spectrum.
# For NICER, they are defined in NormalizeSpectrumNICER.py, and for RXTE, they are defined in .npy

# OUTPUTS (2) of TransmitModel.calculate_transmit_model(): transmit_model array and time_crossing_model array


class TransmitModel:

    mix_N = 0.78
    mix_O = 0.21
    mix_Ar = 0.01

    dE_eff = 0.25  # keV, default step size for the effective transmittance model (NICER)
    N_eff = 10   # Number of effective transmit steps for variable energy band (RXTE)
    dx = 0.0005  # keV, energy step size (in a step of const xsect) to calc probability under the normalized spectrum

    ds_km = 0.5   # km, step size along the telescopic LOS

    def __init__(self, obs_dict, orbit_derived_inputs, eband_derived_inputs):
        # Unpack inputs
        if obs_dict["detector"] == "NICER":
            self.year0 = 2014
        elif obs_dict["detector"] == "RXTE":
            self.year0 = 1994
        self.obs_dict = obs_dict
        self.r_array, self.t_array, self.t0_model_index, self.lat_gp, self.lon_gp = orbit_derived_inputs
        self.e_band, self.bin_size, self.normalized_amplitudes, self.bin_centers = eband_derived_inputs
        self.hc_type = self.obs_dict["hc_type"]
        self.r0_hc = self.r_array[self.t0_model_index]
        self.time_step_r = self.t_array[1] - self.t_array[0]  # time step in r_array and t_array

        self.time_final = 200   # There is a more general way to define this
        self.time_crossing_model = np.arange(0, self.time_final + self.bin_size, self.bin_size, float)

    def calculate_transmit_model(self):

        # determine densities along the LOS at all times during the crossing
        s_dist_max_km = 3000  # [km] ~ 2400 km is half the LOS, do 3000 to be safe
        ds_cm = TransmitModel.ds_km * 10 ** 5  # step size, [cm]
        s_list_km = np.arange(0, s_dist_max_km, TransmitModel.ds_km)   # Same size LOS at every time initially

        density_array, density_tp_list = self.calculate_density_arrays(s_list_km)

        # effective transmittance model
        effective_transmit = np.zeros_like(self.time_crossing_model, float)  # transmittance over time of crossing
        en1_list, en2_list = self.define_energy_integration_lists()   # different for RXTE/NICER
        a = 0
        prob_i = 1/(TransmitModel.N_eff-1)   # TODO: Temporary fix for RXTE testing
        for i in range(len(en1_list)):
            E_mean = np.mean([en1_list[i], en2_list[i]])
            # prob_i = self.calc_spectrum_probability(en1_list[i], en2_list[i])
            a += prob_i
            sigma_i = BCM.get_total_xsect(E_mean, TransmitModel.mix_N, TransmitModel.mix_O, TransmitModel.mix_Ar, 0)
            tau_i = (np.sum(2*density_array, axis=1) + density_tp_list) * sigma_i * ds_cm
            effective_transmit += prob_i * np.exp(-tau_i)
        print(f"a={a} (should be one)")

        return effective_transmit, self.time_crossing_model

    # Function to map out all LOS during the crossing and create an array of densities.
    # OUTPUTS (2): matrix of densities along entire LOS and list of densities at the tangent point
    def calculate_density_arrays(self, s_list_km):
        mid_time_crossing = (self.obs_dict['crossing_time_range'][0] + self.obs_dict['crossing_time_range'][1]) / 2
        mid_datetime_crossing = tools.convert_time(mid_time_crossing, self.year0)

        density_array = np.zeros((len(self.time_crossing_model), len(s_list_km)))

        density_tp_list = np.zeros(len(self.time_crossing_model))

        # +1 below is important to get to tangent point in a setting crossing
        for t_index, time in enumerate(self.time_crossing_model):
            los_points_km = self.line_of_sight(t_index, s_list_km)
            lat_list_deg_pymap, lon_list_deg_pymap, altitude_list_pymap = tools.eci2geodetic_pymap_array(los_points_km, mid_time_crossing, self.year0)
            # Only consider half of the LOS
            tangent_point_index = np.argmin(altitude_list_pymap)
            # print(f"tangent altitude = {altitude_list_pymap[tangent_point_index]}")  # TANGENT ALTITUDE
            los_densities = MSIS.get_pymsis_density(datetime=mid_datetime_crossing,
                                                    lon=self.lon_gp,
                                                    lat=self.lat_gp,
                                                    alts=altitude_list_pymap,
                                                    f107=self.obs_dict["f107"],
                                                    ap=self.obs_dict["ap"],
                                                    version=00)[1]
            density_tp_list[t_index] = los_densities[tangent_point_index]

            los_densities[tangent_point_index:] = 0.0  # only consider densities on half the LOS
            density_array[t_index, :] = los_densities

        return density_array, density_tp_list

    # n km steps along the line of sight
    # Note that in the HorizonCrossings- repo, I have a version of this function where n_list can be an integer
    def line_of_sight(self, time_index, n_list):
        n_column_vec = n_list.reshape((len(n_list), 1))
        starArray = np.ones((len(n_list), 3)) * self.obs_dict['starECI']

        # convert the time_index for time_crossing (bin_size) into an index for r_array (0.01 sec)
        time_index_long = int(time_index / self.time_step_r)
        index_start_setting = self.t0_model_index - int(self.time_final / self.time_step_r)
        # TODO: fix this for setting crossing!!
        if self.hc_type == "rising":
            # start the calculation of the curve at t0 and r0
            return self.r_array[self.t0_model_index + time_index_long] + n_column_vec * starArray
        elif self.hc_type == "setting":
            return self.r_array[index_start_setting + time_index_long] + n_column_vec * starArray

    # FUNCTION: Takes in a range of energies in keV, returns a probability from the normalized spectrum
    # en1 and en2 are located within e_band
    def calc_spectrum_probability(self, en1_kev, en2_kev):
        # Function for normalized amplitude as a function of Energy
        spec = interp1d(self.bin_centers, self.normalized_amplitudes)

        # perform numerical integration between en1 and en2
        prob_in_range = 0  # this is the integral sum of probability. Add up area under the curve
        for left_bound in np.arange(en1_kev, en2_kev, TransmitModel.dx):
            prob_in_range += (spec(left_bound) + spec(left_bound))*(TransmitModel.dx / 2)
        return prob_in_range

    # Integrate the area under the spectrum differently for NICER and RXTE data
    def define_energy_integration_lists(self):
        if self.obs_dict["detector"] == "NICER":
            en1_list = np.arange(self.e_band[0], self.e_band[1], TransmitModel.dE_eff)  # left side of constant sigma steps on spectrum
            en2_list = en1_list + TransmitModel.dE_eff  # right side of constant sigma steps
        else:
            dE = (self.e_band[1] - self.e_band[0])/TransmitModel.N_eff
            en1_list = np.linspace(self.e_band[0], self.e_band[1], TransmitModel.N_eff)
            en1_list = en1_list[:-1]
            en2_list = en1_list + dE
        return en1_list, en2_list

    @classmethod
    def set_ds_km(cls, ds):
        cls.ds_km = ds

    @classmethod
    def set_dE_eff(cls, dE_eff):
        cls.dE_eff = dE_eff

    @classmethod
    def set_N_eff(cls, N_eff):
        cls.N_eff = N_eff
