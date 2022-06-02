# Author: Nathaniel Ruhl
# This script is used to make sure the process of sliding the transmittance curve works for both rising and setting horizon crossings

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import numpy as np
import sys
sys.path.append("/Users/nathanielruhl/PycharmProjects/HorizonCrossings-local/nruhl_final_project")  # add working directory, str(Path(__file__).parents[1])

# import local modules
from AnalyzeCrossing import AnalyzeCrossing

N = 500 # average number of unattenuated counts in data
bin_size = 1
comp_range = [0.01, 0.99] # range of transmittance in which to compare the curves

# This function generates the horizon crossing times and transmittance arrays for both model and data
def generate_crossings(sat, hc_type):
    std = 0.03  # standard deviation of normally-distributed noise
    np.random.seed(3)

    # Define time_model differently if setting or rising HC
    time_model = np.arange(0, sat.time_final, bin_size)
    if hc_type == "setting":
        time_model = np.flip(time_model)

    transmit_model = sat.calc_transmit_curve(time_model)
    transmit_data = transmit_model.copy()

    if hc_type == "rising":
        transmit_data = np.insert(transmit_data, 0, np.zeros(int(100/bin_size)))
        transmit_data = np.append(transmit_data, np.ones(int(100/bin_size)))
        noise_range = np.where(transmit_data>0.05)[0]
        transmit_data[noise_range] += np.random.normal(0, std, len(transmit_data[noise_range]))
    elif hc_type == "setting":
        transmit_data = np.insert(transmit_data, 0, np.ones(int(100/bin_size)))
        transmit_data = np.append(transmit_data, np.zeros(int(100/bin_size)))
        noise_range = np.where(transmit_data>0.05)[0]
        transmit_data[noise_range] += np.random.normal(0, std, len(transmit_data[noise_range]))

    time_data = np.arange(2000-100, 2000+sat.time_final+100, bin_size)
    return time_model, transmit_model, time_data, transmit_data

# This calss executes the curve CurveComparison
class CurveComparison:
    def __init__(self, sat, hc_type):
        self.sat = sat
        self.hc_type = hc_type  # want this to be autonomous
        self.time_model, self.transmit_model, self.time_data, self.transmit_data = generate_crossings(self.sat, self.hc_type)
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
            t0_1 = t50_approx_data + dt50_approx_model
        return t0_1

    # This function slides the model past the data at time intervals of desired_precision and calculates chi_sq
    def locate_t0_step2(self):
        desired_precision = 0.01
        t0_1_index = np.where(self.time_data >= self.t0_1)[0][0]

        # Define the data points in the full crossing time range
        if self.hc_type == "setting":
            time_crossing_data = self.time_data[t0_1_index-len(self.time_model):t0_1_index]
            rate_data = N*self.transmit_data[t0_1_index-len(self.time_model):t0_1_index]
        elif self.hc_type == "rising":
            time_crossing_data = self.time_data[t0_1_index:t0_1_index+len(self.time_model)]
            rate_data = N*self.transmit_data[t0_1_index:t0_1_index+len(self.time_model)]

        # bin size must be greater than 2
        t_start_list = np.arange(int(self.time_data[t0_1_index])-2,
                                 int(self.time_data[t0_1_index])+2,
                                 desired_precision)

        weight_range = np.where((self.transmit_model >= comp_range[0]) & (self.transmit_model <= comp_range[1]))[0]

        chisq_list = np.zeros(len(t_start_list))
        for indx, t0_guess in enumerate(t_start_list):
            # define interpolating function and array for the model
            if self.hc_type == "rising":
                time_crossing_model = np.arange(t0_guess, t0_guess + self.sat.time_final, bin_size)
            elif self.hc_type == "setting":
                time_crossing_model = np.flip(np.arange(t0_guess, t0_guess - self.sat.time_final, -bin_size))

            # Note that however this interpolation is done, the model and data times need to be in the same order
            model_rate_vs_time = interp1d(time_crossing_model, N*self.transmit_model, kind='cubic', fill_value='extrapolate')
            model_rate_interp = model_rate_vs_time(time_crossing_data)
            # List of model values at times where data points are

            # Chi-squared test in weight_range of full curve
            chisq = np.sum(
                (rate_data[weight_range] - model_rate_interp[weight_range]) ** 2 / model_rate_interp[weight_range])
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
        func = interp1d(self.t0_guess_list, self.chisq_list, kind='cubic', fill_value='extrapolate')
        return func(t0)

if __name__ == '__main__':

    sat = AnalyzeCrossing(cb="Earth", H=400, E_kev=2.0)
    comp_obj = CurveComparison(sat, "setting")

    print(f"First guess t0_1 = {comp_obj.t0_1:.2f} sec")
    print(f"Best fit t0_e = {comp_obj.t0_e:.2f} +/- {comp_obj.dt_e:.2f} sec")

    t_hc_setting = np.flip(np.arange(comp_obj.t0_e, comp_obj.t0_e - sat.time_final, -bin_size))
    t_hc_rising = np.arange(comp_obj.t0_e, comp_obj.t0_e + sat.time_final, bin_size)

    # With the numbers that we're using, t0 ~ 2000 (rising) and 2305 (setting)

    plt.figure(1)
    plt.plot(comp_obj.time_data, N*comp_obj.transmit_data, '.', label="Simulated Data")
    plt.plot(t_hc_setting, N*comp_obj.transmit_model, label="Scaled Model")
    plt.ylabel(f"Counts per {bin_size} sec bin")
    plt.xlabel("Time (sec)")

    plt.figure(2)
    plt.title(fr"Minimum: $t_{{0,e}}$ = {comp_obj.t0_e:.2f} +/- {comp_obj.dt_e:.2f} sec")
    plt.plot(comp_obj.t0_guess_list, comp_obj.chisq_list)
    plt.ylabel(r"$\chi^2$")
    plt.xlabel(r"$t_0$ (sec)")

    plt.show()
