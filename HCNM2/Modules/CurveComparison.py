# Author: Nathaniel Ruhl
# This class executes the curve comparison between the model and the data

import numpy as np
from scipy.interpolate import interp1d

# INPUTS:
# obs_dict
# model_and_data_tuple (5): time_model, transmit_model, time_data, rate_data, unattenuated rate (N)
# Note that transmit_data is on a 0-1 scale
# Note that in the length (time duration) of the model and data input arrays may be different


class CurveComparison:

    comp_range = [0.01, 0.99]  # curve comparison transmission range

    def __init__(self, obs_dict, model_and_data_tuple):
        self.obs_dict = obs_dict
        self.hc_type = self.obs_dict["hc_type"]
        self.time_model, self.transmit_model, self.time_data, self.rate_data, self.N = model_and_data_tuple
        self.transmit_data = self.rate_data / self.N
        self.bin_size = self.time_model[1] - self.time_model[0]    # Based on our definition of time_model

        # First step to identify t0
        self.t0_1 = self.locate_t0_step1()
        self.t0_e, self.t0_guess_list, self.chisq_list = self.locate_t0_step2()
        self.dt_e = self.analyze_chisq()

    # This function is used to identify the model time (from t0)
    def time_vs_transmit_model(self, transmit_approx_50):
        frange = np.where((self.transmit_model > 0.01) & (self.transmit_model < 0.99))[0]    # indices for which transmit vs time is 1-1
        time_vs_transmit = interp1d(x=self.transmit_model[frange], y=self.time_model[frange], kind='cubic')
        return time_vs_transmit(transmit_approx_50)

    # first guess of t0 by comparing 50% points
    def locate_t0_step1(self):
        if self.hc_type == "setting":
            index50_approx_data = np.where(self.transmit_data < 0.51)[0][0]
        elif self.hc_type == "rising":
            index50_approx_data = np.where(self.transmit_data > 0.49)[0][0]
        transmit50_approx_data = self.transmit_data[index50_approx_data]
        t50_approx_data = self.time_data[index50_approx_data]

        dt50_approx_model = self.time_vs_transmit_model(transmit50_approx_data)

        if self.hc_type == "rising":
            t0_1 = t50_approx_data - dt50_approx_model
        elif self.hc_type == "setting":
            # Note that this formula for t0_1 is different than the toy model, but necessary here
            t0_1 = t50_approx_data + self.time_model[-1] - dt50_approx_model
        return t0_1

    # This function slides the model past the data at time intervals of desired_precision and calculates chi_sq
    def locate_t0_step2(self):
        desired_precision = 0.01

        t0_1_index = np.where(self.time_data >= self.t0_1)[0][0]

        # Define the data points in the full crossing time range
        if self.hc_type == "setting":
            time_crossing_data = self.time_data[t0_1_index-len(self.time_model)+1:t0_1_index+1]
            rate_data = self.rate_data[t0_1_index-len(self.time_model)+1:t0_1_index+1]
            transmit_data = rate_data / self.N
        elif self.hc_type == "rising":
            time_crossing_data = self.time_data[t0_1_index:t0_1_index+len(self.time_model)]
            rate_data = self.rate_data[t0_1_index:t0_1_index+len(self.time_model)]
            transmit_data = rate_data / self.N

        t_start_list = np.arange(self.t0_1-2,
                                 self.t0_1+2,
                                 desired_precision)

        weight_range = np.where((self.transmit_model >= CurveComparison.comp_range[0]) & (self.transmit_model <= CurveComparison.comp_range[1]))[0]

        chisq_list = np.zeros(len(t_start_list))
        for indx, t0_guess in enumerate(t_start_list):
            # define interpolating function and array for the model
            if self.hc_type == "rising":
                time_crossing_model = t0_guess + self.time_model
            elif self.hc_type == "setting":
                time_crossing_model = np.flip(np.arange(t0_guess, t0_guess - len(self.time_model), -self.bin_size))

            # Note that however this interpolation is done, the model and data times need to be in the same order
            # It is good to setup the interpolating function in the weight range (incase nans in transmit_model)
            rate_model = self.N*self.transmit_model
            model_rate_vs_time = interp1d(time_crossing_model[weight_range], rate_model[weight_range], kind='cubic', fill_value="extrapolate")
            model_rate_interp = model_rate_vs_time(time_crossing_data[weight_range])
            if any(model_rate_interp < 0.0):
                print("Cubic spline went negative")
            # List of model values at times where data points are

            # Chi-squared test in weight_range of full curve (make sure we didn't get any nans here)
            chisq_i = (rate_data[weight_range] - model_rate_interp) ** 2 / model_rate_interp
            chisq_i = np.nan_to_num(chisq_i, nan=0.0)
            chisq = np.sum(chisq_i)
            chisq_list[indx] = chisq

        t0_e = t_start_list[np.argmin(chisq_list)]

        return t0_e, t_start_list, chisq_list

    # Methods to calculate the chisq+1 error
    def analyze_chisq(self):
        upper_t0 = self.bisection_algorithm_chisq(a0=self.t0_e, b0=self.t0_e + 1.5, Y_TOL=10 ** (-5.), NMAX=50,
                                                  chisq_goal=self.chisq_vs_time(self.t0_e)+1)
        lower_t0 = self.bisection_algorithm_chisq(a0=self.t0_e, b0=self.t0_e - 1.5, Y_TOL=10 ** (-5.), NMAX=50,
                                                  chisq_goal=self.chisq_vs_time(self.t0_e)+1)

        # return the larger of the two
        if self.chisq_vs_time(upper_t0) > self.chisq_vs_time(lower_t0):
            chisq_error = upper_t0 - self.t0_e
        else:
            chisq_error = abs(lower_t0 - self.t0_e)

        return chisq_error

    def bisection_algorithm_chisq(self, a0, b0, Y_TOL, NMAX, chisq_goal):
        N = 1
        a = a0
        b = b0
        while N < NMAX:
            c = (a + b) / 2  # midpoint
            if abs(self.chisq_vs_time(c) - chisq_goal) <= Y_TOL:
                return c
            if self.chisq_vs_time(c) > chisq_goal:
                b = c
            elif self.chisq_vs_time(c) < chisq_goal:
                a = c
            N += 1
        print(f'Bisection Not Found to tolerance in NMAX, c = {c}')
        return c

    def chisq_vs_time(self, t0):
        func = interp1d(self.t0_guess_list, self.chisq_list, kind='linear', fill_value='extrapolate')
        return func(t0)

    # class method to change curve comparison range
    @classmethod
    def set_comp_range(cls, comp_range):
        cls.comp_range = comp_range
