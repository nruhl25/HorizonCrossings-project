# Author: Nathaniel Ruhl
# This script reads in the orbit data file for a given observation. It is a parent class of LocatR0hcNav, LocateR0hc2, and FitAtmosphere

import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table

# INPUTS: obs_dict (dict), orbit_model = "mkf", "rossi", or "aster".
# (should make another class if we want to do it for the keplerian orbit)
# Reads r_array, v_array, t_array, directly from data files, defines h_unit
# Creates an interpolating function r(t) that is used in the algorithm to Locate r0_hc

class OrbitModel2:
    def __init__(self, obs_dict, orbit_model):
        self.obs_dict = obs_dict
        self.orbit_model = orbit_model
        self.r_array, self.v_array, self. t_array = self.read_orbit_model()
        self.R_orbit, self.h_unit = self.define_R_orbit_h_unit()

        if self.obs_dict["detector"] == "NICER":
            self.year0 = 2014
        elif self.obs_dict["detector"] == "RXTE":
            self.year0 = 1994

        # Only create the interpolating function for r(t) once, not on every call
        self.rx_interpolator = interp1d(
            self.t_array, self.r_array[:, 0], kind="cubic")
        self.ry_interpolator = interp1d(
            self.t_array, self.r_array[:, 1], kind="cubic")
        self.rz_interpolator = interp1d(
            self.t_array, self.r_array[:, 2], kind="cubic")

    def read_orbit_model(self):
        if self.orbit_model == "mkf":
            r_array, v_array, t_array = OrbitModel2.read_mkf(
                self.obs_dict["mkf_path"])
        elif self.orbit_model == "rossi":
            r_array, v_array, t_array = OrbitModel2.read_rxte_orbit(
                self.obs_dict["rossi_path"])
        # temporary solution until we implement read_aster_orbit()
        elif self.orbit_model == "aster":
            r_array, v_array, t_array = OrbitModel2.read_mkf(
                self.obs_dict["mkf_path"])
        else:
            raise RuntimeError(
                "orbit_model must be either 'mkf', 'rossi', or 'aster'")
        return r_array, v_array, t_array

    # Reads the orbital state from NICER's mkf file
    @staticmethod
    def read_mkf(fn_string):
        tab_mkf = Table.read(fn_string, hdu=1)
        r = np.array(tab_mkf['POSITION'])
        t = np.array(tab_mkf['TIME'])
        v = np.array(tab_mkf['VELOCITY'])
        return r, v, t

    # Reads the orbital state from RXTE file
    @staticmethod
    def read_rxte_orbit(fn_string):
        tab = Table.read(fn_string, hdu=1)
        x = np.array(tab['X']) / 1000.0  # km
        y = np.array(tab['Y']) / 1000.0  # km
        z = np.array(tab['Z']) / 1000.0  # km
        r_array = np.column_stack((x, y, z))
        t_array = np.array(tab['Time'])
        v_x = np.array(tab['Vx']) / 1000.0  # km/s
        v_y = np.array(tab['Vy']) / 1000.0  # km/s
        v_z = np.array(tab['Vz']) / 1000.0  # km/s
        v_array = np.column_stack((v_x, v_y, v_z))  # km/s
        return r_array, v_array, t_array

    # Interpolating function for the LocateR0hc algorithm (only takes in a single time)
    def r(self, t):
        r_x = self.rx_interpolator(t)
        r_y = self.ry_interpolator(t)
        r_z = self.rz_interpolator(t)
        return np.array([r_x, r_y, r_z])

    # Interpolating function used to define v0 at a single time t0
    def v(self, t):
        v_x = interp1d(self.t_array, self.v_array[:, 0], kind="cubic")
        v_y = interp1d(self.t_array, self.v_array[:, 1], kind="cubic")
        v_z = interp1d(self.t_array, self.v_array[:, 2], kind="cubic")
        return np.array([v_x(t), v_y(t), v_z(t)])

    # This method is a general version of the above functio
    # t is the list of times that corresponds to r_array or v_array, the state "x"
    # t_array (which must be an array, is an array of desired times)
    @staticmethod
    def interpolate_state(x, t, t_array):
        x_interpolator = interp1d(t, x[:, 0], kind="cubic")
        y_interpolator = interp1d(t, x[:, 1], kind="cubic")
        z_interpolator = interp1d(t, x[:, 2], kind="cubic")

        x_x = x_interpolator(t_array).reshape((len(t_array), 1))
        x_y = y_interpolator(t_array).reshape((len(t_array), 1))
        x_z = z_interpolator(t_array).reshape((len(t_array), 1))
        x_array = np.hstack((x_x, x_y, x_z))
        return x_array

    # Function used to define R_orbit and h_unit at the ~middle of the crossing time period
    def define_R_orbit_h_unit(self):
        mid_time = (self.obs_dict["crossing_time_range"]
                    [0]+self.obs_dict["crossing_time_range"][1])/2
        mid_time_index = np.where(self.t_array >= mid_time)[0][0]
        R_orbit = np.linalg.norm(self.r_array[mid_time_index])
        h_unit = np.cross(
            self.r_array[mid_time_index], self.v_array[mid_time_index])
        h_unit = h_unit / np.linalg.norm(h_unit)
        return R_orbit, h_unit
