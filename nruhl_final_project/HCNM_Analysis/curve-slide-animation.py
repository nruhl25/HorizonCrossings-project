# Author: Nathaniel Ruhl
# This script makes and animation of the curve comparison. This script should be used whenever working to improve the algorithm

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import numpy as np
import sys
import os
import matplotlib.animation as manimation
# add working directory, str(Path(__file__).parents[1])
sys.path.append(
    "/Users/nathanielruhl/Documents/HorizonCrossings-Summer22/nruhl_final_project/")

# Define the meta data for the movie
# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie test', artist='Nathaniel Ruhl')
# writer = FFMpegWriter(fps=15, metadata=metadata)

# import local modules
from AnalyzeCrossing import AnalyzeCrossing

# Global parameters to be used in the analysis
cb_str = "Venus"  # P2: size of Earth, but a thinner atmosphere
hc_type = "rising"
N0 = 5000  # average number of unattenuated counts in data
E_kev = 1.5
H = 500  # km, orbital altitude
bin_size = 1.0
# range of transmittance in which to compare the curves
comp_range = [0.01, 0.99]

# Parameters involved in generating hc data
std = 0.05  # standard deviation of normally-distributed noise
np.random.seed(3)


# This function generates the horizon crossing times and transmittance arrays for both model and data
def generate_crossings(sat, hc_type):
    # Define time_model differently if setting or rising HC
    time_model = np.arange(0, sat.time_final, bin_size)
    if hc_type == "setting":
        time_model = np.flip(time_model)

    transmit_model = sat.calc_transmit_curve(time_model)

    transmit_data = transmit_model.copy()

    if hc_type == "rising":
        transmit_data = np.insert(
            transmit_data, 0, np.zeros(int(100/bin_size)))
        transmit_data = np.append(transmit_data, np.ones(int(100/bin_size)))
        noise_range = np.where(transmit_data > 0.05)[0]
        transmit_data[noise_range] += np.random.normal(
            0, std, len(transmit_data[noise_range]))
    elif hc_type == "setting":
        transmit_data = np.insert(transmit_data, 0, np.ones(int(100/bin_size)))
        transmit_data = np.append(transmit_data, np.zeros(int(100/bin_size)))
        noise_range = np.where(transmit_data > 0.05)[0]
        transmit_data[noise_range] += np.random.normal(
            0, std, len(transmit_data[noise_range]))

    time_data = np.arange(2000-100, 2000+sat.time_final+100, bin_size)
    return time_model, transmit_model, time_data, transmit_data


# This class executes the curve CurveComparison
class CurveComparison:
    def __init__(self, sat, hc_type, N0):
        self.sat = sat
        self.hc_type = hc_type  # want this to be autonomous
        self.N0 = N0
        self.time_model, self.transmit_model, self.time_data, self.transmit_data = generate_crossings(
            self.sat, self.hc_type)
        # First step to identify t0
        self.t0_1 = self.locate_t0_step1()
        self.t0_new = self.locate_t0_alternative()
        # import time
        # time.sleep(5)
        self.t0_e, self.t0_guess_list, self.chisq_list = self.locate_t0_step2()
        self.dt_e = self.analyze_chisq()

    # This function is used to identify the model time (from t0)
    def time_vs_transmit_model(self, transmit_approx_50):
        frange = np.where((self.transmit_model > 0.01) & (self.transmit_model < 0.99))[
            0]    # indices for which transmit vs time is 1-1
        time_vs_transmit = interp1d(
            x=self.transmit_model[frange], y=self.time_model[frange], kind='cubic')
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
            time_crossing_data = self.time_data[t0_1_index -
                                                len(self.time_model):t0_1_index]
            rate_data = self.N0 * \
                self.transmit_data[t0_1_index-len(self.time_model):t0_1_index]
            transmit_data = self.transmit_data[t0_1_index -
                                               len(self.time_model):t0_1_index]
        elif self.hc_type == "rising":
            time_crossing_data = self.time_data[t0_1_index:t0_1_index+len(
                self.time_model)]
            rate_data = self.N0 * \
                self.transmit_data[t0_1_index:t0_1_index+len(self.time_model)]
            transmit_data = self.transmit_data[t0_1_index -
                                               len(self.time_model):t0_1_index]

        t_start_list = np.arange(self.t0_1-1,
                                 self.t0_1+1,
                                 desired_precision)
        
        weight_range = np.where((self.transmit_model >= comp_range[0]) & (self.transmit_model <= comp_range[1]))[0]

        chisq_list = np.zeros(len(t_start_list))
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("Horizon Crossing Curve Comparison")
        fig.supxlabel("Time (sec)")
        # with writer.saving(fig, "writer_test.mp4", 100):
        for indx, t0_guess in enumerate(t_start_list):
            ax[0].clear()
            # define interpolating function and array for the model
            if self.hc_type == "rising":
                time_crossing_model = np.arange(
                    t0_guess, t0_guess + self.sat.time_final, bin_size)
            elif self.hc_type == "setting":
                time_crossing_model = np.flip(
                    np.arange(t0_guess, t0_guess - self.sat.time_final, -bin_size))
            else:
                continue
            
            # Note that however this interpolation is done, the model and data times need to be in the same order
            model_rate_vs_time = interp1d(
                time_crossing_model, self.N0*self.transmit_model, kind='linear')
            model_rate_interp = model_rate_vs_time(
                time_crossing_data[weight_range])

            # List of model values at times where data points are

            if any(model_rate_interp <= 0):
                print("Cubic spline went negative")

            # Chi-squared test in weight_range of full curve
            chisq = np.sum(
                (rate_data[weight_range] - model_rate_interp) ** 2 / model_rate_interp)
            chisq_list[indx] = chisq

            if indx % 4 == 0:
                ax[0].plot(time_crossing_data[weight_range], model_rate_interp, 'b-')
                ax[0].plot(time_crossing_data[weight_range], rate_data[weight_range], 'kx')
                ax[1].plot(t0_guess, chisq, 'r.')
                ax[0].set_xlim([min(time_crossing_data[weight_range]-1), max(time_crossing_data[weight_range])+1])
                ax[0].set_ylim([0, max(rate_data[weight_range])+10])
                ax[0].set_ylabel("Counts/sec")
                ax[1].set_ylabel(r"$\chi^2$")
                plt.pause(0.01)
                # writer.grab_frame()

        t0_e = t_start_list[np.argmin(chisq_list)]
        print(t0_e)
        plt.show()

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
        func = interp1d(self.t0_guess_list, self.chisq_list,
                        kind='cubic', fill_value="extrapolate")
        return func(t0)

    def locate_t0_alternative(self):
        # Same method as locate_t0_step1(), but for multiple points in the transmission curve
        cross_range = np.where((self.transmit_data >= 0.1)
                               & (self.transmit_data <= 0.8))[0]
        t_data_range = self.time_data[cross_range]

        dt_model_list = self.time_vs_transmit_model(
            self.transmit_data[cross_range])

        if self.hc_type == "rising":
            t0_new_list = t_data_range - dt_model_list
        elif self.hc_type == "setting":
            t0_new_list = t_data_range + dt_model_list
        return np.mean(t0_new_list)

if __name__ == "__main__":

    sat = AnalyzeCrossing(cb_str, H, E_kev)
    comp_obj = CurveComparison(sat, hc_type, N0=N0)
    print(comp_obj.dt_e)
