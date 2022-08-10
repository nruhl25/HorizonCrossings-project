# Author: Nathaniel Ruhl
# This script contains functions to "fit the atmosphere" to tanh() before doing navigation

from Modules.OrbitModel2 import OrbitModel2

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

class FitAtmosphere(OrbitModel2):
    def __init__(self, obs_dict, orbit_model, r0_obj, rate_data, time_data, unattenuated_rate, y50_predicted):
        OrbitModel2.__init__(self, obs_dict, orbit_model)
        # Unpack r0_obj (LocateR0hcNav):
        self.s_unit = r0_obj.s_unit
        self.h0_ref = r0_obj.h0_ref
        self.y0_ref = r0_obj.y0_ref   # R_b in Breck 2023 paper
        self.t0_model = r0_obj.t0_model
        self.A = r0_obj.A
        self.y50_predicted = y50_predicted
        self.h50_predicted = self.y50_predicted - self.y0_ref

        # Unpack other inputs:
        self.rate_data = rate_data
        self.time_data = time_data
        self.hc_type = self.obs_dict["hc_type"]
        self.unattenuated_rate = unattenuated_rate

        # Derived values (note the convention that "measured" is in the shorter time window of time_crossing)
        self.h_measured, self.n_measured, self.rate_measured, self.t_measured = self.calcTangentAltitudes(time_crossing=200)
        self.transmit_measured = self.rate_measured/self.unattenuated_rate  # used to define valid fit range

        self.a_fit, self.b_fit, self.c_fit, self.d_fit, self.h1, self.h2 = self.fit_transmit_vs_alt()
        self.dt = self.get_dt()  # navigational measurement
        self.t0_e = self.t0_model - self.dt   # if applied at t0
        # h1 and h2 is the valid altitude range for the arctangent fit
        # note that c_fit is a tangent altitude, not a radial distance to the tangent point

    # This function calculates y(t) that corresponds to the binned data
    # time_crossing (int) is the total expected duration for which to calculate y(t)
    def calcTangentAltitudes(self, time_crossing):
        # times that corrspond to binned data
        if self.hc_type == "rising":
            start_index = np.where(self.time_data > self.t0_model)[0][0]
            t_measured = self.time_data[start_index] + np.arange(0.0, time_crossing, 1.0)
            rate_measured = self.rate_data[start_index:start_index+time_crossing]
        elif self.hc_type == "setting":
            start_index = np.where(self.time_data < self.t0_model)[0][-1]
            t_measured = np.flip(np.arange(
                self.time_data[start_index], self.time_data[start_index]-time_crossing, -1))
            # Shorten rate_data to rate_measured
            rate_measured = self.rate_data[start_index-time_crossing+1:start_index+1]
        
        # Fill in the tangent altitude and LOS distance during the crossing time period
        # km, altitude above self.y0_ref
        h_measured = np.zeros_like(t_measured)
        # km, distance along LOS to closest approach
        n_measured = np.zeros_like(t_measured)
        for i in range(len(h_measured)):
            h_measured[i], n_measured[i] = self.h(t_measured[i])
        
        return h_measured, n_measured, rate_measured, t_measured

    # This function calculates the distance of the tangent point telescopic line of sight at a time t. It minimizes f(t, n) over n to define y(t).
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

    # This function returns the geocentric altitude of any point along the LOS with index n (any time t) above self.y0_ref
    def f(self, t, n):
        eci_vec = self.r(t) + n*self.s_unit
        alt_n = np.linalg.norm(eci_vec) - self.y0_ref
        return alt_n

    def fit_transmit_vs_alt(self):
        # Note that w1 can't be defined if h0_ref is too large
        comp_range = np.where((self.transmit_measured >= 0.01) & (self.transmit_measured <= 0.99))[0]
        # if self.hc_type == "rising":
        #     w1 = np.where(self.transmit_measured >= 0.01)[0][0]  # can't go to zero
        #     w2 = np.where(self.transmit_measured >= 0.99)[0][0]+1
        # elif self.hc_type == "setting":
        #     w1 = np.where(self.transmit_measured <= 0.99)[0][0]
        #     w2 = np.where(self.transmit_measured <= 0.01)[0][-1]+1
        #     # can't go to zero (0.03 because of obs 50099, not 0.01)
        transmit_measured = self.transmit_measured[comp_range]   #[w1:w2]
        # specify valid range for fit
        h_measured = self.h_measured[comp_range]   #[w1:w2]
        popt, pcov = curve_fit(self.transmit_vs_h, h_measured, transmit_measured, p0=[
                               0.5, 1/50, 100, 0.5])
        a, b, c, d = popt

        return a, b, c, d, h_measured[0], h_measured[-1]

    # hyperbolic tangent fit
    def transmit_vs_h(self, h, a, b, c, d):
        return a*np.tanh(b*(h-c))+d

    def get_dt(self):
        # define interpolating function between time and tangent altitude
        t_vs_h = interp1d(self.h_measured, self.t_measured, "cubic")
        dt = t_vs_h(self.c_fit) - t_vs_h(self.h50_predicted)
        return dt
