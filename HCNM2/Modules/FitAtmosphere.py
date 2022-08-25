# Author: Nathaniel Ruhl
# This script contains functions to "fit the atmosphere" to tanh() before doing navigation. The class fits tanh() to the altitude curve rate

from Modules.OrbitModel2 import OrbitModel2
from Modules.ADM import predict_y50
from Modules.tools import minimize_scalar, root_scalar

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import operator

# self.comp_range is very helpful and indexes to self.time_meeasured, self, transmit_measuredd, and self.h_measured

class FitAtmosphere(OrbitModel2):
    def __init__(self, obs_dict, orbit_model, r0_obj, rate_data, time_data, unattenuated_rate, e_band_kev):
        OrbitModel2.__init__(self, obs_dict, orbit_model)
        # Unpack r0_obj (LocateR0hcNav):
        self.s_unit = r0_obj.s_unit
        self.h0_ref = r0_obj.h0_ref
        self.y0_ref = r0_obj.y0_ref   # R_b in Breck 2023 paper
        self.t0_model = r0_obj.t0_model
        self.A = r0_obj.A
        self.e_band_kev = e_band_kev
        self.y50_predicted = predict_y50(self.obs_dict, self.e_band_kev)   # TODO: This only has the ADM for a single energy band! (h50_fit and h50_predict won't be close)
        self.h50_predicted = self.y50_predicted - self.y0_ref

        # Unpack other inputs:
        self.rate_data = rate_data
        self.time_data = time_data
        self.hc_type = self.obs_dict["hc_type"]
        self.unattenuated_rate = unattenuated_rate

        # Derived values used in fit (note the convention that "measured" is in the shorter time window of time_crossing)
        self.h_measured, self.n_measured, self.rate_measured, self.time_measured = self.calcTangentAltitudes(time_crossing=200)
        self.transmit_measured = self.rate_measured/self.unattenuated_rate  # used to define valid fit range

        # Make measurements of y50 (and t50) via the tanh() fit
        # note that c_fit is a tangent altitude, not a radial distance to the tangent point
        self.popt, self.pcov, self.comp_range = self.fit_transmit_vs_alt()
        self.a_fit, self.b_fit, self.c_fit, self.d_fit = self.popt
        self.h50_fit = np.arctanh((0.5-self.d_fit)/self.a_fit)/self.b_fit+self.c_fit
        self.y50_fit = self.h50_fit + self.y0_ref # (ATMOSPHERIC MEASUREMENT)
        self.t50_fit = self.get_t50_fit()  # measured time at 50% transmission point (NAV MEASUREMENT)

        # Error analysis for y50 and t50 from propogation of tanh() fit parameters
        self.var_y50 = self.get_var_y50()  # varience of y50 (for atmospheric filter)
        self.dy50 = np.sqrt(self.var_y50)   # standard deviation of y50
        self.var_t50 = self.get_var_t50()  # varience of t50
        self.dt50 = np.sqrt(self.var_t50)  # standard deviation of t50

        self.dt50_slide = self.get_dt50_slide()

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

    # This function is called from self.fit_transmit_vs_alt() and returns the horizon crossing index range for the curve fitting.
    # I don't know if this is better than the [w1:w2] method, but it seems to work well
    def get_comp_range(self):
        comp_range = np.where((self.transmit_measured <= 0.99) & (
            self.transmit_measured >= 0.01))[0]  # initial comp_range
        # Only get the range with the most consecutive ascending/descending values
        num_dict = {}   # starting values are keys, number of sequential elements that follow are columns
        for start_indx in range(len(comp_range)-1):
            i = start_indx + 1
            while (comp_range[start_indx] + (i-start_indx) == comp_range[i]) & (i < len(comp_range)-1):
                i += 1
            num_dict[comp_range[start_indx]] = i-start_indx

        sorted_tuples = sorted(num_dict.items(), key=operator.itemgetter(1))
        # keys are the number of consecutive entries
        sorted_dict = {v: k for k, v in sorted_tuples}
        max_consec = max(list(num_dict.values()))
        comp_range_better = sorted_dict[max_consec] + np.arange(0, max_consec+1, 1)
        return comp_range_better

    def fit_transmit_vs_alt(self):

        comp_range = self.get_comp_range()

        a_guess = 0.5
        b_guess = 1/50    # we need to know how much tangent altitude is spanned from 1% to 99% (should be an under-estimate, use the highest energy to determine the under-estimate)
        d_guess = 0.5
        c_guess = 25 - np.arctanh((0.5-d_guess)/a_guess)/b_guess  # (must under-estimate y50 and thus c also)
        transmit_measured = self.transmit_measured[comp_range]
        # specify valid range for fit
        h_measured = self.h_measured[comp_range]
        popt, pcov = curve_fit(self.transmit_vs_h, h_measured, transmit_measured, p0=[
                               a_guess, b_guess, c_guess, d_guess], bounds=(0,[1, 1, 500, 1]))

        return popt, pcov, comp_range

    # hyperbolic tangent fit
    def transmit_vs_h(self, h, a, b, c, d):
        return a*np.tanh(b*(h-c))+d

    # convert from the y50_measurement to the t50 measurement
    def get_t50_fit(self):
        t_vs_h = interp1d(self.h_measured, self.time_measured, "cubic")
        return t_vs_h(self.h50_fit)

    def get_var_y50(self):
        a, b, c, d = self.popt
        partials_y50 = np.array([-(0.5 - d)/(a**2*b*(1 - (0.5 - d)**2/a**2)),
                             -np.arctanh((0.5 - d)/a)/b**2,
                             1,
                             -1/(a*b*(1 - (0.5 - d)**2/a**2))])  # vector of partial derivatives of y_50
        var_y50 = np.dot(partials_y50.T, np.dot(self.pcov, partials_y50))
        ###### Try only including certain fit parameters in the error #####
        # partials_short = np.array([-np.arctanh((0.5 - d)/a)/b**2, 1])  # b and c
        # pcov_short = np.zeros((2,2))
        # pcov_short[0,0] = self.pcov[0,0]
        # pcov_short[1,1] = self.pcov[1,1]
        # var_y50 = np.dot(partials_short.T, np.dot(pcov_short, partials_short))
        return var_y50

    # midpoint derivative to get the varience of t50 based on the varience of y50
    def get_var_t50(self):
        # define interpolating function between time and tangent altitude
        t_vs_h = interp1d(self.h_measured, self.time_measured, "cubic")
        dy = 1e-5   # typically used for double precision machines
        dt_dy = (t_vs_h(self.h50_fit+0.5*dy) - t_vs_h(self.h50_fit-0.5*dy))/dy
        var_t50 = (dt_dy**2)*self.var_y50
        return var_t50

    def plot_tanh_fit(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.h_measured+self.y0_ref, self.rate_measured/self.unattenuated_rate, ".")
        h_model = np.linspace(min(self.h_measured), max(self.h_measured), 1000)

        plt.title(f"{self.e_band_kev[0]}-{self.e_band_kev[1]} keV")
        plt.plot(h_model+self.y0_ref, self.transmit_vs_h(h_model,
                                                            self.a_fit, self.b_fit, self.c_fit, self.d_fit))
        plt.ylabel("Transmittance, $T$")
        plt.xlabel("Tangent radius, $y$ (km)")
        plt.show()
        return 0

    # Newton/secant method to determine t50 directly from time and transmission data (for comparison)
    # Note that this doesn't always work, especially with low SNR
    def get_t50_newton(self):
        if self.hc_type == "rising":
            t50_guess_index = np.where(self.transmit_measured[self.comp_range] > 0.5)[0][0]
        elif self.hc_type == "setting":
            t50_guess_index = np.where(self.transmit_measured[self.comp_range] < 0.5)[0][0]
        t50_guess = self.time_measured[self.comp_range][t50_guess_index]
        transmit_vs_time = interp1d(
            self.time_measured[self.comp_range], self.transmit_measured[self.comp_range], kind='linear')

        f = lambda t: transmit_vs_time(t) - 0.5   # function for finding the root
        delta = 1
        t50 = t50_guess
        tlast = t50_guess + 0.05
        num_iter = 0
        while abs(delta) > 1e-5:
            b = f(t50)
            blast = f(tlast)
            m = (b-blast)/(t50-tlast)
            delta = b/m
            tlast = t50
            t50 -= delta
            num_iter += 1

        return t50

    def slide_t50(self):
        '''This function slides the model curve past the data for the chisq analysis.'''
        time_vs_h = interp1d(self.h_measured, self.time_measured, kind='cubic')
        h_vs_time = interp1d(self.time_measured, self.h_measured, kind='cubic')
        t50_slide_list = np.arange(self.t50_fit-0.5, self.t50_fit+0.5, 0.005)
        h50_slide_list = h_vs_time(t50_slide_list)
        chisq_list = []
        for t50_i, h50_i in zip(t50_slide_list, h50_slide_list):
            c_i = h50_i - (1/self.b_fit)*np.arctanh((0.5-self.d_fit)/self.a_fit)
            transmit_model = self.transmit_vs_h(self.h_measured, self.a_fit, self.b_fit, c_i, self.d_fit)
            rate_model = transmit_model*self.unattenuated_rate
            # Note that there is a different comp_range during the curve slide, as the model could go negative
            chisq_terms = (self.rate_measured - rate_model)**2/(rate_model)
            if self.obs_dict['obsID'] == 50099:
                comp_range = np.where((transmit_model > 0.03)
                                    & (transmit_model < 0.99))[0]
            else:
                comp_range = np.where((transmit_model > 0.03)
                                    & (transmit_model < 0.99))[0]
            chisq_list.append(np.sum(chisq_terms[comp_range]))
        return t50_slide_list, np.array(chisq_list)

    def get_dt50_slide(self):
        '''Seamus should update this algorithm with his new method. I implemented this here so we can get the measurement uncertainties, but we should definitely improve this algorithm'''
        t50_slide_list, chisq_list = self.slide_t50()
        f = interp1d(t50_slide_list, np.array(chisq_list), kind='cubic')
        t50 = minimize_scalar(f, x0=np.mean([t50_slide_list[0], t50_slide_list[-1]]))
        
        # Solve chisq+1 on left/right
        f_left = lambda t: f(t) - f(t50) + 1
        t_left = root_scalar(f_left, x0=t50, x0_last = t50-0.05)
        f_right = lambda t: f(t) - f(t50) - 1
        t_right = root_scalar(f_right, x0=t50, x0_last=t50+0.05)

        dt_right = t_right - t50
        dt_left = t50 - t_left
        if dt_right >= dt_left:
            dt50 = dt_right
        else:
            dt50 = dt_left
        return dt50
