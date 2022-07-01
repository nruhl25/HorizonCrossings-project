# Author: Nathaniel Ruhl
# This script plots the maximum out of plane angle for an Earth satelitte at different altitudes (Central Body is defined as a global variable in psi_solver, if we want to switch it)

import numpy as np
import matplotlib.pyplot as plt

# Import local modules
from psi_solver import *  # GLOBAL VARIABLE R_planet DEFINED IN PSI_SOLVER.PY
R_planet = 6378.0  # over-ride psi_solver since the simulation was done with this

# geometrical formula
def theta_max(H):
    theta = np.arctan2(R_planet, np.sqrt((R_planet+H)**2-R_planet**2))
    return np.rad2deg(theta)

def calc_psiMax(H, d_psi):
    R_orbit = R_planet + H
    psi_list = np.arange(0, 80, d_psi)
    for psi in psi_list:
        r0_hc, s_unit, num_iter = rotate_and_find_r0hc(psi, R_orbit)
        if np.isnan(r0_hc).any():
            psi_break = psi
            print(f"H={H}km")
            print(f'psi_break = {psi_break} deg with {num_iter} iterations')
            break
        else:
            continue

    psi_err = d_psi/2
    psi_max = psi_break - psi_err
    return psi_max

def main(d_psi=1):
    altitude_list = np.arange(400, 1100, 100, float)  # km
    psi_err_list = (d_psi/2)*np.ones_like(altitude_list, float)   # deg
    psi_max_list = np.zeros_like(psi_err_list)   # deg
    for i, H in enumerate(altitude_list):
        psi_max_list[i] = calc_psiMax(H, d_psi)

    # np.save("theta_altitude.npy", np.column_stack((psi_max_list, psi_err_list, altitude_list)))

    # Make the line for the model
    alt_list_model = np.arange(400, 1001, 1, float)  # km

    plt.figure(1)
    plt.errorbar(x=altitude_list, y=psi_max_list, yerr=psi_err_list, label="Algorithm divergence")
    plt.plot(alt_list_model, theta_max(alt_list_model), label=r"$\theta_{max}$ model")
    plt.ylabel("Maximumum out-of-plane angle (deg)")
    plt.xlabel("Orbital altitude (km)")
    plt.ylim([min(psi_max_list)-5, max(psi_max_list)+5])
    plt.grid()
    plt.legend()

    # Dimensionless plot
    Rstar = (R_planet + altitude_list)/R_planet
    Rstar_model = (R_planet + alt_list_model)/R_planet
    plt.figure(2)
    plt.errorbar(x=Rstar, y=psi_max_list, yerr=psi_err_list)
    plt.plot(Rstar_model, theta_max(alt_list_model), label=r"$\theta_{max}$ model")
    plt.ylabel("Maximumum out-of-plane angle (deg)")
    plt.xlabel(r"$R_{orbit}/R_{planet}$")
    plt.ylim([0, max(psi_max_list)+5])
    plt.grid()

    plt.show()

    return 0

# Used to re-plot if we already have the simulation results saved to a file
def plot_theta_max_data():
    alt_list_model = np.arange(300, 3001, 1, float)  # km
    arr = np.load("/Users/nathanielruhl/Documents/HorizonCrossings-Summer22/nruhl_final_project/out-of-plane-geometry/theta_altitude.npy")
    psi_max_list = arr[:,0]
    psi_err_list = arr[:,1]
    altitude_list = arr[:,2]
    plt.figure(1)
    plt.plot(alt_list_model, theta_max(alt_list_model), label=r"$\theta_{max}$ model")
    plt.errorbar(x=altitude_list, y=psi_max_list, yerr=psi_err_list, label="Algorithm divergence", fmt='+')
    plt.ylabel("Maximumum out-of-plane angle (deg)")
    plt.xlabel("Orbital altitude (km)")
    plt.ylim([min(psi_max_list)-5, max(psi_max_list)+5])

    plt.figure(2)
    plt.title("Errors")
    plt.plot(altitude_list, psi_max_list - theta_max(altitude_list))

    print(f"Max difference = {np.max(psi_max_list - theta_max(altitude_list))} +/- {psi_err_list[0]} deg")
    plt.grid()
    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    import time
    start_time = time.time()
    # main()
    plot_theta_max_data()
    print("--- %s seconds ---" % (time.time() - start_time))


