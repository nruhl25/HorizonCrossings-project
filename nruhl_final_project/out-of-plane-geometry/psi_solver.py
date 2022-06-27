# Author: Nathaniel Ruhl
# This script introduces a new algorith to "locate r0_hc" that uses non-linear root solving. Define desired central body on line 12 and run the main() function.

import numpy as np
import matplotlib.pyplot as plt

# Define constants
G = 6.6743*10**(-11)    # Nm^2/kg^2, Gravitational constant
R_earth = 6378.0  # km
M_earth = 5.972 * 10 ** 24  # kg, mass of Earth
M_mars = 0.64169 * 10 ** 24  # kg
R_mars = 3396.2   # km, equatorial radius
M_mercury = 0.3301e24  # kg
R_mercury = 2440.5   # km, equatorial radius
R_moon = 1738.1  # km
M_moon = 0.07346e24   # kg
################ DEFINE CENTRAL BODY ################
R_planet = R_earth
M_planet = M_earth
hc_type = "rising"   # or "setting"
h_unit = np.array([0, 0, 1])  # (aka we're alread in the perifocal frame)
#####################################################

# Function to project the source onto the plane of the orbit
def proj_on_orbit(r_source, h_unit):
    r_prime_source = r_source - h_unit * \
        np.dot(r_source, h_unit)   # project on orbit plane
    r_prime_source = r_prime_source / \
        np.linalg.norm(r_prime_source)  # normalize the vector
    return r_prime_source

# position of satelite in orbital model (or an interpolating function)
def r(t, R_orbit):
    T = np.sqrt(4*np.pi**2/(G*M_planet) * (R_orbit*10**3)**3)
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

# Function to minimize when identifying r0
def f(t, s_unit, R_orbit):
    # define the los
    A_2d = np.sqrt(R_orbit ** 2 - R_planet ** 2)  # distance of half los for in-plane crossing
    n_max = 1.1*A_2d  # km, max distance along LOS to look for grazing, a 3d los is always shorter than a 2d
    n_list = np.arange(0, n_max, 0.1)   # 0.1 km steps (a little larger than A_2d (incase the orbital radius changed)
    n_column_vec = n_list.reshape((len(n_list), 1))

    starArray = np.ones((len(n_list), 3)) * s_unit
    los = r(t, R_orbit) + n_column_vec * starArray
    p_mag = np.linalg.norm(los, axis=1)
    # A_3d=0.1*np.argmin(p_mag) A_3d should in theory always be smaller than A_2d
    alt_tp = np.min(p_mag)  # Altitude of tangent point, km (this eqn is slightly different for allipsoid planet in which case R_planet is an array that depends on los polar angles)
    return alt_tp - R_planet

# This function returns r0_hc for an arbitrary out-of-plane angle, psi (deg).
def find_r0hc(s_unit, R_orbit):
    # Derived values
    psi_deg = np.rad2deg((np.pi/2)-np.arccos(np.dot(h_unit, s_unit)))
    T = np.sqrt(4*np.pi**2/(G*M_planet) * (R_orbit*10**3)**3)
    t_orbit = np.arange(0, T, 1)   # must be defined for initial r0_2d guess

    # Define r0_2d
    s_proj = proj_on_orbit(s_unit, h_unit)
    # Use the 2d formulas to guess where r0 may be
    if hc_type == "rising":
        g_unit_proj = np.cross(s_proj, h_unit)
    elif hc_type == "setting":
        g_unit_proj = np.cross(h_unit, s_proj)

    A_2d = np.sqrt(R_orbit ** 2 - R_planet ** 2)
    r0_2d = R_planet * g_unit_proj - A_2d * s_proj

    dr = np.linalg.norm(r(t_orbit, R_orbit) - r0_2d, axis=1)   # list of errors from r0_2d
    t1_index = np.argmin(dr)
    t1 = t_orbit[t1_index]  # t_0,guess

    # Newton's method to minimize f(t)
    t = t1  # initial guess
    t_last = t1 - 1  # initial guess for secant method

    b_last = 2*R_orbit  # initialization to enter for loop
    graze_tolerance = 1e-3 # 75 m, altitude tolerance for identifying the graze point (max difference between  geodetic and geocentric altitude)
    num_iter = 1
    while(abs(b_last)>graze_tolerance):
        b = f(t, s_unit, R_orbit)
        m = (f(t, s_unit, R_orbit) - f(t_last, s_unit, R_orbit))/(t-t_last)
        if b is np.nan or m is np.nan or abs(b_last) < abs(b):
            # b must be monotonically decreasing
            # No solution found (r0_hc will have a 'nan' in it)
            break
        delta = b/m
        b_last = b
        t_last = t
        t -= delta
        num_iter += 1

    # If we broke out of the loop, r0_hc will include a 'nan'
    if b is np.nan or m is np.nan or abs(b_last) < abs(b):
        r0_hc = np.array([np.nan, np.nan, np.nan])
    else:
        r0_hc = r(t, R_orbit)
    # print(f"psi = {psi_deg} deg")
    # print(f"{num_iter} iterations")
    # print(f"r0_2d = {r0_2d}")
    # print(f"r0_model1 = {r(t1, R_orbit)}")
    # print(f"r0_hc = {r0_hc}")
    # print("-------------------")
    return r0_hc, num_iter

# This function rotates the in-plane source s1 = np.array([0, 1, 0]) (perifocal fram)
#  about the x-axis by psi_deg and returns r0_hc
def rotate_and_find_r0hc(psi_deg, R_orbit, s1=np.array([0, 1, 0])):
    # Rotate the source position
    psi = np.deg2rad(psi_deg)   # radians
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(psi), -np.sin(psi)],
                    [0, np.sin(psi), np.cos(psi)]])   # Principle rotation about x axis

    s_unit = np.dot(R_x, s1)

    r0_hc, num_iter = find_r0hc(s_unit, R_orbit)
    return r0_hc, s_unit, num_iter

# This function calculates and plots r0_hc for a single orbit up to psi_break
# Input: H is the orbital alitude above R_planet (km)
# Input: d_psi is the step size (deg) in out-of-plane angle
def main(H, d_psi):
    R_orbit = R_planet + H
    # Plot the initial orbit
    T = np.sqrt(4*np.pi**2/(G*M_planet) * (R_orbit*10**3)**3)
    t_orbit = np.arange(0, T, 1)   # must be defined to create orbit_vec
    orbit_vec = r(t_orbit, R_orbit)
    plt.figure()
    plt.title(f"{hc_type} horizon crossing at H = {H} km above Earth")
    plt.scatter(orbit_vec[:, 0], orbit_vec[:, 1], s=1)

    # find the r0 value for psi_list
    psi_list = np.arange(0, 80, d_psi)  # max seems to be 69 for the ISS orbit
    for psi in psi_list:
        r0_hc, s_unit, num_iter = rotate_and_find_r0hc(psi, R_orbit)
        if np.isnan(r0_hc).any():
            psi_break = psi
            print(f"H={H}km")
            print(f'psi_break = {psi_break} deg with {num_iter} iterations')
            break
        else:
            plt.scatter(r0_hc[0], r0_hc[1], label=fr"$\psi = ${psi}$^\circ$")
            continue

    psi_err = d_psi/2
    psi_max = psi_break - psi_err
    print(f"Therefore, psi_max={psi_max}+/-{psi_err} deg")
    plt.plot([], [], 'k', label=fr"$\psi_{{max}}$={psi_max}$\pm${psi_err}$^\circ$")
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    # Consider a single Earth orbit at H=420 km:
    main(H=420, d_psi=1)


    # Code to verify Seamus's out-of-plane angle formula:
    # First, find s_unit corresponding to theta_max (coordinate conversion)
    # R_orbit = R_planet + 580
    # s1 = np.array([-1, 0], float)
    # phi = np.arccos(R_planet/R_orbit)
    # T = np.array([[-np.cos(phi), np.sin(phi)],[-np.sin(phi), -np.cos(phi)]])

    # s2 = np.dot(T, s1)
    #
    # s_unit = np.array([s2[1], 0, s2[0]])   # Back to ECI-like vector (go a little closer to the plane to see convergence)
    # s_unit = s_unit / np.linalg.norm(s_unit)

    # r0_hc, num_iter = find_r0hc(s_unit, R_orbit)