# Author: Nathaniel Ruhl
# This is a new script to calculate the tranmittance model

# INPUTS:
# r02_obj (object of LocateR0hc2)
# eband_derived_inputs = (e_band, bin_size, normalized_amplitudes, bin_centers)
# normalized_amplitudes and bin_centers: Information from source spectrum.
# For NICER, they are defined in NormalizeSpectrumNICER.py, and for RXTE, they are defined in a .npy file

# self.time_crossing_model_met, self.time_crossing_model self.transmit_model defined from these inputs

import numpy as np
from scipy.interpolate import interp1d

import Modules.tools as tools
from Modules.xsects import BCM  # X-ray cross sections
import Modules.MSIS as MSIS  # densisty model

class TransmitModel2:
    mix_N = 0.78
    mix_O = 0.21
    mix_Ar = 0.01

    # keV, default step size for the effective transmittance model (NICER)
    dE_eff = 0.20
    # Number of effective transmit steps for variable size energy band (RXTE)
    N_eff = 4
    # keV, energy step size (in a step of const xsect) to calc probability under the normalized spectrum
    dx = 0.005

    ds_km = 0.5   # km, step size along the telescopic LOS

    pymsis_version = 2

    def __init__(self, r02_obj, eband_derived_inputs):
        # Unpack inputs (don't want a subclass that will re-run the init function)
        self.e_band, self.bin_size, self.normalized_amplitudes, self.bin_centers = eband_derived_inputs
        self.obs_dict = r02_obj.obs_dict
        self.hc_type = self.obs_dict["hc_type"]
        self.s_unit = self.obs_dict["starECI"]
        self.r_array = r02_obj.r_array
        self.t_array = r02_obj.t_array
        self.t0_model = r02_obj.t0_model
        self.year0 = r02_obj.year0
        self.lat_gp = r02_obj.lat_gp
        self.lon_gp = r02_obj.lon_gp
        self.s_dist_max_km = r02_obj.n_max  # max distance across LOS to look for grazing

        # Only create the interpolating function for r(t) once, not on every call
        self.rx_interpolator = interp1d(
            self.t_array, self.r_array[:, 0], kind="cubic")
        self.ry_interpolator = interp1d(
            self.t_array, self.r_array[:, 1], kind="cubic")
        self.rz_interpolator = interp1d(
            self.t_array, self.r_array[:, 2], kind="cubic")


        # Calculate the transmittance model from r0_hc. This is the time array we will use for r(t)
        # t0 with altitude 0 must be the first element in the list
        self.time_final = 175  # TODO: This definition must be more generalizeable
        if self.hc_type == "rising":
            self.time_crossing_model_met = np.arange(0, self.time_final, self.bin_size, float) + self.t0_model
        elif self.hc_type == "setting":
            time_decreasing = np.arange(
                self.t0_model, self.t0_model - self.time_final, -self.bin_size, float)
            self.time_crossing_model_met = np.flip(time_decreasing)
        
        # time_model is [0:time_final], wheras time_model_met is in seconds of MET
        self.time_crossing_model = self.time_crossing_model_met - self.time_crossing_model_met[0]
        self.transmit_model = self.calculate_transmit_model()

    def calculate_transmit_model(self):
        # determine densities along the LOS at all times during the crossing
        ds_cm = TransmitModel2.ds_km * 10 ** 5  # step size, [cm]
        # Same size LOS at every time initially
        s_list_km = np.arange(
            0, self.s_dist_max_km, TransmitModel2.ds_km)
        density_array, density_tp_list = self.calculate_density_arrays(
            s_list_km)

        # effective transmittance model
        # transmittance over time of crossing
        effective_transmit = np.zeros_like(self.time_crossing_model_met, float)
        # different for RXTE/NICER
        en1_list, en2_list = self.define_energy_integration_lists()
        a = 0
        # prob_i = 1/TransmitModel.N_eff   # Temporary fix for integration testing
        for i in range(len(en1_list)):
            E_mean = np.mean([en1_list[i], en2_list[i]])
            prob_i = self.calc_spectrum_probability(en1_list[i], en2_list[i])
            a += prob_i
            sigma_i = BCM.get_total_xsect(
                E_mean, TransmitModel2.mix_N, TransmitModel2.mix_O, TransmitModel2.mix_Ar, 0)
            tau_i = (np.sum(2*density_array, axis=1) +
                     density_tp_list) * sigma_i * ds_cm
            effective_transmit += prob_i * np.exp(-tau_i)
        print(f"a={a} (should be exactly one)")
        # This may occur for setting crossing
        effective_transmit = np.nan_to_num(effective_transmit, posinf=np.nan)

        return effective_transmit

    # Function to map out all LOS during the crossing and create an array of densities.
    # OUTPUTS (2): matrix of densities along entire LOS and list of densities at the tangent point
    def calculate_density_arrays(self, s_list_km):
        mid_time_crossing = (
            self.obs_dict['crossing_time_range'][0] + self.obs_dict['crossing_time_range'][1]) / 2
        mid_datetime_crossing = tools.convert_time(
            mid_time_crossing, self.year0)

        density_array = np.zeros(
            (len(self.time_crossing_model_met), len(s_list_km)))

        density_tp_list = np.zeros(len(self.time_crossing_model_met))

        for t_index, t in enumerate(self.time_crossing_model_met):
            los_points_km = self.line_of_sight(t, s_list_km)
            los_mag_list = np.sqrt(
                los_points_km[:, 0] ** 2 + los_points_km[:, 1] ** 2 + los_points_km[:, 2] ** 2)
            # Calculate altitudes corresponding to points on the LOS:
            # geocentric latitudes
            polar_angles = np.arccos(los_points_km[:, 2] / los_mag_list)
            earth_points = tools.point_on_earth_azimuth_polar(
                np.zeros_like(polar_angles), polar_angles)
            earth_radius_list = np.sqrt(
                earth_points[:, 0] ** 2 + earth_points[:, 1] ** 2 + earth_points[:, 2] ** 2)
            altitude_list = los_mag_list - earth_radius_list

            # Only consider half of the LOS
            tangent_point_index = np.argmin(altitude_list)
            # print(f"tangent altitude = {altitude_list[tangent_point_index]}")  # TANGENT ALTITUDE
            los_densities = MSIS.get_pymsis_density(datetime=mid_datetime_crossing,
                                                    lon=self.lon_gp,
                                                    lat=self.lat_gp,
                                                    alts=altitude_list,
                                                    f107=self.obs_dict["f107"],
                                                    ap=self.obs_dict["Ap"],
                                                    version=TransmitModel2.pymsis_version)[1]
            density_tp_list[t_index] = los_densities[tangent_point_index]

            # only consider densities on half the LOS
            los_densities[tangent_point_index:] = 0.0
            density_array[t_index, :] = los_densities

        return density_array, density_tp_list

    # n km steps along the line of sight
    # t = element of self.time_crossing_model_met
    def line_of_sight(self, t, n_list):
        n_column_vec = n_list.reshape((len(n_list), 1))
        starArray = np.ones((len(n_list), 3)) * self.s_unit
        return self.r(t) + n_column_vec * starArray

    # Interpolating function for the LocateR0hc algorithm (only takes in a single time)
    def r(self, t):
        r_x = self.rx_interpolator(t)
        r_y = self.ry_interpolator(t)
        r_z = self.rz_interpolator(t)
        return np.array([r_x, r_y, r_z])

    # FUNCTION: Takes in a range of energies in keV, returns a probability from the normalized spectrum
    # en1 and en2 are located within e_band
    def calc_spectrum_probability(self, en1_kev, en2_kev):
        # Function for normalized amplitude as a function of Energy
        spec = interp1d(self.bin_centers, self.normalized_amplitudes)

        # perform numerical integration between en1 and en2
        prob_in_range = 0  # this is the integral sum of probability. Add up area under the curve
        for left_bound in np.arange(en1_kev, en2_kev, TransmitModel2.dx):
            prob_in_range += (spec(left_bound) + spec(left_bound +
                                                      TransmitModel2.dx))*(TransmitModel2.dx / 2)
        return prob_in_range

    # Integrate the area under the spectrum differently for NICER and RXTE data
    def define_energy_integration_lists(self):
        if self.obs_dict["detector"] == "NICER":
            # left side of constant sigma steps on spectrum
            en1_list = np.arange(
                self.e_band[0], self.e_band[1], TransmitModel2.dE_eff)
            en2_list = en1_list + TransmitModel2.dE_eff  # right side of constant sigma steps
        else:
            dE = (self.e_band[1] - self.e_band[0])/TransmitModel2.N_eff
            en1_list = np.arange(self.e_band[0], self.e_band[1], dE)
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

    @classmethod
    def set_pymsis_version(cls, version):
        if version == 00 or version == 2:
            cls.pymsis_version = version
        else:
            print("pymsis version must be either 2 or 00. 2 is being used as default")
        
        






