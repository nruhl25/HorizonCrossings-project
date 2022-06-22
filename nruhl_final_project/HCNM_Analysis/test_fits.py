# This script test function fits for the precision vs altitude analysis

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

dr_list = np.load("dr_list.npy")
altitude_list = np.load("altitude_list.npy")

def invLog(x, a, b, c):
    return a/(np.log(b*x)) + c

popt, pcov = curve_fit(invLog, altitude_list, dr_list)
a, b, c = popt

alts = np.linspace(min(altitude_list), max(altitude_list), 1000)

plt.plot(altitude_list, dr_list)
plt.plot(alts, invLog(alts, *popt), label=fr"$\delta r_e$ = {a:.2f}/(ln({b:.3f}H))+{c:.2f}")
plt.ylabel(r"$\delta r_e$ (km)")
plt.xlabel(r"Orbital altitude, $H$ (km)")
plt.legend()
plt.show()
