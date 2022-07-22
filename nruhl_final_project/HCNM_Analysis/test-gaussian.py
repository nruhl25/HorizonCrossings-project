# Author: Nathaniel Ruhl
# This script is used to test a gaussin fit on the chi squared vs t0_guess_list data

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/nathanielruhl/Documents/HorizonCrossings-Summer22/nruhl_final_project/")

data = np.load("HCNM_Analysis/t0-chisq-array.npy")
t0_guess_list = data[:,0]
chisq_list = data[:,1]
sym_range = np.arange(np.argmin(chisq_list)-4, np.argmin(chisq_list)+5, 1)   # This could be defined better in the future...

# Can use bisection to better define this
chisq_list = chisq_list[sym_range]
t0_guess_list = t0_guess_list[sym_range]

def gaussian(t, a, b, c, k):
    return k + a*np.exp(-((t-b)**2)/(2*c**2))

popt, pcov = curve_fit(gaussian, t0_guess_list, chisq_list)

t = np.linspace(t0_guess_list[0], t0_guess_list[-1], 1000)

plt.plot(t0_guess_list, chisq_list)
plt.plot(t, gaussian(t, *popt))

plt.show()
