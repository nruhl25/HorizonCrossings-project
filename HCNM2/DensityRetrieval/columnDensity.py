# Author: Nathaniel Ruhl

# In this script, I start with Transmittance and Tangent Altitude data from a 1-2 keV horizon crossing of the Crab Nebula, determine the column densities at each tangent altitude via Newton's method, then fit a set of smoothed exponentials to the data points. After that, I re-construct the transmittance model and see if it matches with the data

# Data Files for 1-2 keV of the HC of the Crab Nebula:
# timeRate.npy is the full binned data during the crossing
# altT0.npy contains the tangent altitudes and a list of times after t0 for which tangent points are calculated.

from xsects import BCM

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

timeTransmit = np.load("/Users/nathanielruhl/Documents/HorizonCrossings-Summer22/nruhl_final_project/DensityMeasurement/timeTransmit.npy")
time_data = timeTransmit[:,0]
transmit_data = timeTransmit[:,1]
timeAlt = np.load(
    "/Users/nathanielruhl/Documents/HorizonCrossings-Summer22/nruhl_final_project/DensityMeasurement/timeAlt.npy")
time_crossing_model = timeAlt[:,0]
h_list = timeAlt[:,1]

crossing_range = np.where((time_data>=time_crossing_model[0])&(time_data<=time_crossing_model[-1]))[0]
time_data = time_data[crossing_range]
transmit_data = transmit_data[crossing_range]

# 1) Since the transmit data doesnt correspond to time_crossing_model, we first need to interpolate h_list
# We can avoid this is the future by by definining time_crossing_model differently in another script so that instead of t0_model+[0:175:1], it is t0_model+[dt1:175+dt1:1] where dt1 = time_data[0] - t0_model... but the interpolation is fine for this example
h_vs_time = interp1d(time_crossing_model, h_list, "cubic")
h_data = h_vs_time(time_data)

#2) Now, we have transmit_data and h_data measured at time_data for 1-2 keV.
M_molar_air = 28.97 # [g/mol]
N_a = 6.0221408e23  # avogadro's number
M_air = M_molar_air/N_a   # molecular weight of air [g/molecule]
sigma_tot = BCM.get_total_xsect(1.5, mix_N=0.78, mix_O=0.21, mix_Ar=0.01, mix_C=0.0)*M_air  # cm^2
print(BCM.get_total_xsect(1.5, mix_N=0.78, mix_O=0.21, mix_Ar=0.01, mix_C=0.0))

# function to minimize when solving N (column density)
def f(F, N):
    return F - np.exp(-sigma_tot*N)

# Measure column densities, N, via Newton (Note that F = transmit, finishes with about 2 iterations each)
w1 = np.where(transmit_data>=0.01)[0][0]
w2 = np.where(transmit_data >= 0.99)[0][0] # cannot be greater than one!!!
# Lists of measured values, length of weight range
# h_measured = []
# N_measured = []

# for h, F in zip(h_data[w1:w2], transmit_data[w1:w2]):
#     N = 1e10  # initial guess (smaller side seems to always work)
#     delta = 1.0
#     tol = 1e-5
#     num_iter = 0
#     while(delta>tol and num_iter < 10):
#         delta = f(F, N)/f_prime(N)
#         N -= delta
#         num_iter += 1
#     h_measured.append(h)
#     N_measured.append(N)
h_measured = h_data[w1:w2]
N_measured = -np.log(transmit_data[w1:w2])/sigma_tot

# h_i is an extra arg, not a fit parameter
def N(h, alpha, beta, h_i):
    return alpha * np.exp(-beta*(h-h_i))

M = 2  # Number of points before point i and after point i+1 to include in the fit for smoothing
# Parameters on N fit
alphas = np.zeros_like(N_measured)
betas = np.zeros_like(N_measured)
for i in range(len(N_measured)-1):
    p1 = i - M
    plast = i + 1 + M
    if p1 < 0:
        p1 = 0
        plast = p1 + 1 + M
    elif plast > len(N_measured):
        plast = len(N_measured)
    print(f"i={i}")
    print(f"p1={p1}")
    print(f"plast={plast}")
    
    # Curve fit
    print(f"{len(h_measured[p1:plast+1])} points in fit")
    popt, pcov = curve_fit(lambda h, alpha, beta: N(
        h, alpha, beta, h_measured[i]), h_measured[p1:plast+1], N_measured[p1:plast+1], p0=[1e20, 1/50])
    alphas[i] = popt[0]
    betas[i] = popt[1]
### PLOT RESULTS ####

plt.figure()
plt.title("Column Density vs Tangent Altitude")
plt.ylabel(fr'Total tangential column density (cm$^{{-2}}$)')
plt.xlabel("Tangent Altitude (km)")
plt.plot(h_measured, N_measured, ".", label="Data")
for i in range(len(N_measured)-1):
    h_spl = np.linspace(h_measured[i], h_measured[i+1], 1000)
    N_spl = N(h_spl, alphas[i], betas[i], h_measured[i])
    plt.plot(h_spl, N_spl)
plt.yscale("log")
plt.legend()

plt.figure()
# re-cacluclate transmittance curve
plt.ylabel("Transmittance")
plt.xlabel("Tangent Altitude (km)")
plt.plot(h_data, transmit_data, ".")
for i in range(len(N_measured)-1):
    h_spl = np.linspace(h_measured[i], h_measured[i+1], 1000)
    N_spl = N(h_spl, alphas[i], betas[i], h_measured[i])
    plt.plot(h_spl, np.exp(-sigma_tot*N_spl))
plt.show()

