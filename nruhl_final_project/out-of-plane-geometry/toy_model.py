# Author: Nathaniel Ruhl
# This script is used to visualize the horizon crossing and to plot psiMax against R_orbit/R_planet. In this script, we set R_planet = 1 so we can interpret R_orbit as the ratio.

# This produces an interesting plot, but I don't think it's completely correct

import numpy as np
import matplotlib.pyplot as plt

########## Global variables ###########
R_planet = 1.0
hc_type = "rising"   # or "setting"
h_unit = np.array([0, 0, 1])  # (aka we're alread in the perifocal frame)

# Although time doesn't really matter, we want to keep the relationship between variables the same
G = 1.0    # Nm^2/kg^2, Gravitational constant
M_planet = 1.0  # kg, mass of planet
########################################

# position of satelite in orbital model (or an interpolating function)

def rModel(t, R_orbit, T):
    omega = 2*np.pi/T
    if isinstance(t, np.ndarray):
        x = R_orbit*np.cos(omega*t)
        y = R_orbit*np.sin(omega*t)
        z = np.zeros_like(x)
        return np.column_stack((x, y, z))
    else:
        # t is a scalar
        x = R_orbit*np.cos(omega*t)
        y = R_orbit*np.sin(omega*t)
        z = 0.0
        return np.array([x, y, z])

# Function to project the source onto the plane of the orbit
def proj_on_orbit(r_source, h_unit):
    r_prime_source = r_source - h_unit * \
        np.dot(r_source, h_unit)   # project on orbit plane
    r_prime_source = r_prime_source / \
        np.linalg.norm(r_prime_source)  # normalize the vector
    return r_prime_source


# Function to minimize when identifying r0
# Below, some global variables relevant to function
# km, max distance along LOS to look for grazing (should be fully general)
n_max = 1.5
n_list = np.arange(0, n_max, 0.1)   # 0.001 km steps
n_column_vec = n_list.reshape((len(n_list), 1))
def fModel(t, s_unit, R_orbit, T):
    # define the los
    starArray = np.ones((len(n_list), 3)) * s_unit
    los = rModel(t, R_orbit, T) + n_column_vec * starArray
    p_mag = np.linalg.norm(los, axis=1)
    # Altitude of tangent point, km (this eqn is slightly different for allipsoid planet in which case R_planet is an array that depends on los polar angles)
    alt_tp = np.min(p_mag)
    return alt_tp - R_planet

# This function returns r0_hc for an arbitrary out-of-plane angle, psi (deg).
def find_r0hc(psi_deg, R_orbit, T):
    # Rotate the source position
    psi = np.deg2rad(psi_deg)   # radians
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(psi), -np.sin(psi)],
                    [0, np.sin(psi), np.cos(psi)]])   # Principle rotation about x axis
    s1 = np.array([0, 1, 0])   # in-plane source on y-axis (of perifocal frame)

    s_unit = np.dot(R_x, s1)

    # Define r0_2d
    s_proj = proj_on_orbit(s_unit, h_unit)
    # Use the 2d formulas to guess where r0 may be
    if hc_type == "rising":
        g_unit_proj = np.cross(s_proj, h_unit)
    elif hc_type == "setting":
        g_unit_proj = np.cross(h_unit, s_proj)

    A_2d = np.sqrt(R_orbit ** 2 - R_planet ** 2)
    r0_2d = R_planet * g_unit_proj - A_2d * s_proj

    # list of errors from r0_2d
    t_orbit = np.arange(0, T, 0.1)
    dr = np.linalg.norm(rModel(t_orbit, R_orbit, T) - r0_2d, axis=1)
    t1_index = np.argmin(dr)
    t1 = t_orbit[t1_index]

    # Newton's method to minimize f(t)
    t = t1  # initial guess
    t_last = t1 - 1  # initial guess for secant method

    b_last = 2*R_orbit  # initialization to enter for loop
    graze_tolerance = 1e-2  # 10 m, altitude tolerance for identifying the graze point
    num_iter = 1
    while(abs(b_last) > graze_tolerance):
        b = fModel(t, s_unit, R_orbit, T)
        m = (fModel(t, s_unit, R_orbit, T) - fModel(t_last, s_unit, R_orbit, T))/(t-t_last)
        if b is np.nan or m is np.nan or abs(b_last) < abs(b):
            # b must be monotonically decreasing
            break
        delta = b/m
        b_last = b
        t_last = t
        t -= delta
        num_iter += 1

    # If we broke out of the loop, r0_hc must include a 'nan'
    if b is np.nan or m is np.nan or abs(b_last) < abs(b):
        r0_hc = np.array([np.nan, np.nan, np.nan])
    else:
        r0_hc = rModel(t, R_orbit, T)
    return r0_hc, s_unit, num_iter

# Function to calculate psi_max
def calc_psiMax(R_orbit, d_psi, T):

    psi_list = np.arange(0, 90 + d_psi, d_psi)
    for psi in psi_list:
        r0_hc, s_unit, num_iter = find_r0hc(psi, R_orbit, T)
        if np.isnan(r0_hc).any():
            psi_break = psi
            print(f"R_orbit/R_planet={R_orbit}")
            print(f'psi_break = {psi_break} deg with {num_iter} iterations')
            break
        else:
            continue
    
    # This becomes a problem when we're too far away from the planet
    if psi == 90:
        psi_break = np.nan
        psi_err = d_psi/2
        psi_max = psi_break - psi_err
    else:
        psi_err = d_psi/2
        psi_max = psi_break - psi_err
    return psi_max

# MAIN VISUALIZATION FUNCTION
def visualize_crossing(R_orbit=1.05):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot the orbit circle
    T=100
    t_orbit = np.arange(0, T, 0.1)
    orbit_vec = rModel(t_orbit, R_orbit, T)
    plt.title(f"{hc_type} horizon crossing at R_orbit/R_planet = {R_orbit}")
    ax.scatter3D(orbit_vec[:, 0], orbit_vec[:, 1], orbit_vec[:,2], s=1)

    psi_list = np.arange(0, 80, 10)
    for psi_deg in psi_list:
        r0_hc, s_unit, num_iter = find_r0hc(psi_deg, R_orbit)
        ax.quiver(*r0_hc, *s_unit, label=fr"$\psi$={psi_deg}$^\circ$")
    plt.legend()
    plt.show()
    return 0

# MAIN FUNCTION TO PLOT R_orbit/R_planet relationship
def plot_general_relationship(d_psi = 1):
    z_list = np.arange(R_planet+0.01, 2.5*R_planet, 0.01)  # dimensionless ratio
    psiMax_list = []  # deg
    psiErr_list = (d_psi/2)*np.ones_like(z_list)  # deg
    for z in z_list:
        T = np.sqrt(4*np.pi**2/(G*M_planet)) * z**3
        psiMax_list.append(calc_psiMax(z, d_psi, T))
    plt.errorbar(z_list, psiMax_list, yerr=psiErr_list)
    plt.ylabel("Maximum out-of-plane angle (deg)")
    plt.xlabel(r"$R_{orbit}/R_{planet}$")
    plt.show()
    return 0

if __name__ == '__main__':
    # visualize_crossing()
    plot_general_relationship()
