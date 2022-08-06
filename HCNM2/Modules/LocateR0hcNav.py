# Author: Nathaniel Ruhl
# This script proviveds an alternative version of LocateR0hc2.py, but is specific for the navigational problem and ignores the hard rock Earth.

from Modules.OrbitModel2 import OrbitModel2  # reads in orbital data 
import Modules.constants as constants
import Modules.tools as tools

import numpy as np

# Class LocateR0Nav is a subclass of OrbitModel2
# INPUTS: obs_dict (dict), orbit_model = "mkf", "rossi", or "aster", h_ref: altitude above constants.R_EARTH of reference sphere, used for Locating r0.

class LocateR0hcNav(OrbitModel2):
    def __init__(self, obs_dict, orbit_model, h0_ref):
        OrbitModel2.__init__(self, obs_dict, orbit_model)
        self.s_unit = self.obs_dict["starECI"]
        self.h0_ref = h0_ref
        self.y0_ref = self.h0_ref + constants.R_EARTH

        # Length of the LOS to the grazing point
        self.A = np.sqrt(self.R_orbit ** 2 - constants.R_EARTH ** 2)

        # Identify r0_hc, t0_model (the time at r0_hc predicted by input model), psi_deg (out-of-plane angle), graze_point vector
        self.r0_hc, self.t0_model, self.psi_deg, self.graze_point = self.find_r0hc()
        # can define lat_gp and lon_gp based on graze_point
        self.v0_model = self.v(self.t0_model)

    # This function returns the altitude of any point along the LOS with index n (any time t)
    def f(self, t, n):
        eci_vec = self.r(t) + n*self.s_unit
        alt_n = np.linalg.norm(eci_vec) - self.y0_ref
        return alt_n

    # This function calculates the tangent altitude of the telescopic line of sight at a time t (used in find_r0hc()). It minimizes f(t, n) over n to define h(t).
    def h(self, t):
        # Initialize Newton's method for optimization
        dn = 1e-2  # km, step for numerical derivatives
        n = self.A - 1
        n_accuracy = 1e-9  # km = 1 mm along los
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
        return alt_tp, n

    # Main function to identify r0_hc and a couple other things
    def find_r0hc(self):
        # Derived values
        psi_deg = np.rad2deg((np.pi/2)-np.arccos(np.dot(self.h_unit, self.s_unit)))
        # must be defined for initial r0_2d guess (could search entire orbit if we didn't know the range)
        t_orbit = np.arange(
            self.obs_dict["crossing_time_range"][0], self.obs_dict["crossing_time_range"][1], 1)
        r_orbit = self.interpolate_state(self.r_array, self.t_array, t_orbit)

        ### Define r0_2d ###
        s_proj = tools.proj_on_orbit(self.s_unit, self.h_unit)
        # Use the 2d formulas to guess where r0 may be
        if self.obs_dict["hc_type"] == "rising":
            g_unit_proj = np.cross(s_proj, self.h_unit)
        elif self.obs_dict["hc_type"] == "setting":
            g_unit_proj = np.cross(self.h_unit, s_proj)
        
        r0_2d = self.y0_ref * g_unit_proj - self.A * s_proj

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
        t_tolerance = 1e-6  # sec
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
            alt_tp, n_graze = self.h(t)
            graze_point = self.r(t) + n_graze*self.s_unit
            # print(f"alt_tp={alt_tp} km")
            # print(f"n_graze={n_graze} km")
        return r0_hc, t, psi_deg, graze_point
