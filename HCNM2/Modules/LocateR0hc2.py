# Author: Nathaniel Ruhl
# This is the new script to read in an orbit model and define r0_hc

import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table

import Modules.constants as constants
import Modules.tools as tools
import Modules.ell_tools as ell_tools  # "ellipsoid" tools

# INPUTS: obs_dict (dict), orbit_model = "mkf", "rossi", or "aster".
# (should make another class if we want to do it for the keplerian orbit)
# Reads r_array, v_array, t_array, directly from data files, defines h_unit
# Creates an interpolating function r(t) that is used in the algorithm to Locate r0_hc

class OrbitModel2:
    def __init__(self, obs_dict, orbit_model, use_geodetic=False):
        self.obs_dict = obs_dict
        self.orbit_model = orbit_model
        self.use_geodetic = use_geodetic  # default is to use geodetic to save run time
        self.r_array, self.v_array, self. t_array = self.read_orbit_model()
        self.R_orbit, self.h_unit = self.define_R_orbit_h_unit()

        if self.obs_dict["detector"] == "NICER":
            self.year0 = 2014
        elif self.obs_dict["detector"] == "RXTE":
            self.year0 = 1994

        # Only create the interpolating function for r(t) once, not on every call
        self.rx_interpolator = interp1d(self.t_array, self.r_array[:, 0], kind="cubic")
        self.ry_interpolator = interp1d(self.t_array, self.r_array[:, 1], kind="cubic")
        self.rz_interpolator = interp1d(self.t_array, self.r_array[:, 2], kind="cubic")

    def read_orbit_model(self):
        if self.orbit_model == "mkf":
            r_array, v_array, t_array = OrbitModel2.read_mkf(self.obs_dict["mkf_path"])
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


class LocateR0hc2(OrbitModel2):
    def __init__(self, obs_dict, orbit_model):
        OrbitModel2.__init__(self, obs_dict, orbit_model)
        self.s_unit = self.obs_dict["starECI"]

        # Other variables unique to the LocateR0_hc algorithm:
        # distance of half los for in-plane crossing, max distance along LOS to look for grazing since a 3d los to grazing is always shorter than a 2d
        self.A_2d = np.sqrt(self.R_orbit ** 2 - constants.b ** 2)
        self.n_max = self.A_2d + 200
        # 0.1 km steps (a little larger than A_2d (incase the orbital radius changed)
        self.n_list = np.arange(0, self.n_max, 0.1)
        self.n_column_vec = self.n_list.reshape((len(self.n_list), 1))
        self.starArray = np.ones((len(self.n_list), 3)) * self.s_unit

        # Identify r0_hc, t0_model (the time at r0_hc predicted by input model), psi_deg (out-of-plane angle), graze_point vector
        self.r0_hc, self.t0_model, self.psi_deg, self.graze_point = self.find_r0hc()
        self.lat_gp, self.lon_gp, alt_gp = tools.eci2geodetic_pymap_vector(
            self.graze_point, self.t0_model, self.year0)
        self.v0_model = self.v(self.t0_model)
        print(f"TransmitModel2: alt_gp2 = {alt_gp} km")

    # This function returns the altitude of any point along the LOS with index n
    def f(self, t, n):
        # TODO: Be careful that this is in eci in the future when we have an arbitrary orbit, not perifocal
        eci_vec = self.r(t) + n*self.s_unit
        alt_n = ell_tools.eci2llh(eci_vec)[2]
        return alt_n

    # This function calculates the tangent altitude of the telescopic line of sight at a time t (used in find_r0hc()). It minimizes f(t, n) over n to define h(t).
    def h(self, t):
        # 1) Make an inital guess of the tangent point A_3d, using vectorization
        # distance of half los for in-plane crossing
        A_2d = np.sqrt(self.R_orbit ** 2 - constants.a ** 2)
        # km, max distance along LOS to look for grazing, a 3d los is always shorter than a 2d
        n_list = np.arange(0, 1.1*A_2d, 0.1)
        n_column_vec = n_list.reshape((len(n_list), 1))
        starArray = np.ones((len(n_list), 3)) * self.s_unit
        los_array = self.r(t) + n_column_vec * starArray
        # List of magnitudes of poins along the LOS
        p_mag_list = np.linalg.norm(los_array, axis=1)
        A_3d = 0.1*np.argmin(p_mag_list)

        # Initialize Newton's method for optimization
        dn = 1e-2  # km, step for numerical derivatives
        n = A_3d - 1
        n_accuracy = 1e-6  # km = 1 mm along los
        delta = 100.0
        num_iter = 0
        while abs(delta) > n_accuracy and num_iter < 10:
            b = self.f(t, n)
            b_m = self.f(t, n-dn)   # b "minus"
            b_p = self.f(t, n+dn)   # b "plus"
            g = (b_p - b_m)/(2*dn)   # derivative
            gg = (b_m - 2*b + b_p)/(dn**2)  # second derivative
            delta = g/gg
            n -= delta
            num_iter += 1
        # km, must re-define based on updated n
        alt_tp = self.f(t, n)

        return alt_tp, A_3d

    # Main function to identify r0_hc and a couple other things
    def find_r0hc(self):
        # Derived values
        psi_deg = np.rad2deg((np.pi/2)-np.arccos(np.dot(self.h_unit, self.s_unit)))
        T = np.sqrt(4*np.pi**2/(constants.G*constants.M_EARTH) * (self.R_orbit*10**3)**3)
        # must be defined for initial r0_2d guess (could search entire orbit if we didn't know the range)
        t_orbit = np.arange(self.obs_dict["crossing_time_range"][0], self.obs_dict["crossing_time_range"][1], 1)
        r_orbit = self.interpolate_state(self.r_array, self.t_array, t_orbit)

        ### Define r0_2d ###
        s_proj = tools.proj_on_orbit(self.s_unit, self.h_unit)
        # Use the 2d formulas to guess where r0 may be
        if self.obs_dict["hc_type"] == "rising":
            g_unit_proj = np.cross(s_proj, self.h_unit)
        elif self.obs_dict["hc_type"] == "setting":
            g_unit_proj = np.cross(self.h_unit, s_proj)

        A_2d = np.sqrt(self.R_orbit ** 2 - constants.b ** 2)
        r0_2d = constants.b * g_unit_proj - A_2d * s_proj

        # list of errors from r0_2d
        dr = np.linalg.norm(r_orbit - r0_2d, axis=1)
        t1_index = np.argmin(dr)
        t1 = t_orbit[t1_index]  # t_0,guess

        ### Newton's method to minimize f(t) ###
        # initialization to enter for loop
        t = t1  # initial guess
        t_last = t1 - 1  # initial guess for secant method
        b_last = self.h(t_last)[0]   # must be initialized before for loop
        delta = 1.0  # sec time error
        t_tolerance = 1e-5  # sec  # corresponds to about 1e-8 km graze tolerance
        num_iter = 0
        num_iter = 1
        while abs(delta) > t_tolerance and num_iter < 15:
            b = self.h(t)[0]
            m = (b - b_last)/(t-t_last)
            if b is np.nan or m is np.nan:
                # No solution found (r0_hc will have a 'nan' in it)
                break
            b_last = b
            t_last = t
            delta = b/m
            t -= delta
            num_iter += 1

        # If we broke out of the loop, r0_hc will include a 'nan'
        if b is np.nan or m is np.nan or num_iter >= 25:
            r0_hc = np.array([np.nan, np.nan, np.nan])
        else:
            r0_hc = self.r(t)
            # Identify the graze point vector
            A_3d = self.h(t)[1]
            graze_point = self.r(t) + A_3d*self.s_unit
        return r0_hc, t, psi_deg, graze_point
