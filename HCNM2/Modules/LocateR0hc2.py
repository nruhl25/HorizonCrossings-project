# Author: Nathaniel Ruhl
# This is the new script to read in an orbit model and define r0_hc

import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table

from Modules.OrbitModel2 import OrbitModel2
import Modules.constants as constants
import Modules.tools as tools
import Modules.ell_tools as ell_tools  # "ellipsoid" tools

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
