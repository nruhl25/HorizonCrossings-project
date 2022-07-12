# Author: Nathaniel Ruhl
# This file explores the dimensionless parameters identified with the Buckingham Pi theorem

import numpy as np
import matplotlib.pyplot as plt

from AnalyzeCrossing import AnalyzeCrossing

def main(SAT):
    time_array = np.arange(0, SAT.time_final+1, 1)
    transmit = np.zeros_like(time_array)
    for i, t in enumerate(time_array):
        transmit[i] = np.exp(-SAT.tau_gauss(t, 50))

    # Dimensionless parameters
    h = SAT.tan_alt(time_array)
    h_star = h/SAT.scale_height
    lambda_star = SAT.rho0*SAT.sigma*h
    Rstar = SAT.rho0*SAT.sigma*SAT.scale_height
    plt.figure(1)
    plt.plot(h_star, lambda_star, label=fr"{SAT.cb} satellite, $R^*=${Rstar}")
    plt.ylabel(r"$\lambda^*$")
    plt.xlabel(r"$h^*$")
    plt.legend()
    plt.figure(2)
    plt.plot(h_star, transmit, label=f"L={SAT.scale_height} km")
    plt.ylabel("Transmittance")
    plt.xlabel("$h^*$")
    plt.legend()
    plt.figure(3)
    plt.plot(h, transmit, label=f"L={SAT.scale_height} km")
    plt.ylabel("Transmittance")
    plt.xlabel("h")
    return 0

if __name__ == '__main__':
    ES = AnalyzeCrossing(cb="Earth", H=420)
    scale_height_list = [6, 7, 8, 9, 10, 11, 12]
    for height in scale_height_list:
        ES.scale_height = height
        main(ES)
    plt.show()

