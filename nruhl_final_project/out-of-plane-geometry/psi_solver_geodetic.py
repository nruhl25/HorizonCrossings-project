# Author: Nathaniel Ruhl
# This script applies the "locate r0_hc" algorithm to an ellipsoid earth and uses the geodetic latitude instead of geocentric latitude.

import numpy as np
import matplotlib.pyplot as plt

from psi_solver_ellipsoid import point_on_earth_azimuth_polar

# Global variables
hc_type = "rising"   # or "setting"
h_unit = np.array([0, 0, 1])  # (aka we're already in the perifocal frame)

M_planet = 5.972 * 10 ** 24  # kg, mass of Earth
G = 6.6743*10**(-11)    # Nm^2/kg^2, Gravitational constant

# Oblate Earth Model - WGS84 datum
a_e = 6378.137  # [km] semi-major axis
b_e = 6378.137  # [km] semi-major axis
c_e = 6356.752  # [km] semi-minor axis
e = 0.08182   # Eccentricity from Bate et al.

# This function converts an eci coordinate to lat/lon/height using the geocentric to geodetic altitude conversion (Clynch, 2008)
# INPUTS: eci_vec (km)
# OUTPUTS: lat (rad), lon (deg) (not from greenwhich), height (km)

def eci2llh(eci_vec):
    x = eci_vec[0]
    y = eci_vec[1]
    z = eci_vec[2]
    # 1) Compute earth-centered radius of point [km]
    r = np.linalg.norm(eci_vec)
    p = np.sqrt(x ** 2 + y ** 2)
    lon = np.arctan2(y, x)

    # 2) Start computational loop assuming phi_now = geocentric latitude
    phi_now = np.arctan2(z, p)
    h = calc_h(phi_now, eci_vec)  # became nan... why?
    h_last = 1000000.0  # initialize to enter for loop
    tolerance = 1e-4   # [km] --> 10 cm
    while abs(h - h_last) > tolerance:
        Rn = calc_Rn(phi_now)
        phi_next = np.arctan((z/p)*(1-(Rn/(Rn+h))*e**2)**(-1.))
        h_last = h
        h = calc_h(phi_next, eci_vec)
        phi_now = phi_next

    lat = phi_now

    return lat, lon, h

# Helper functions for geocentric to geodetic conversion
# INPUT: geodetic lat (rad)
# OUTPUT: Rn (km)


def calc_Rn(phi):
    return a_e / np.sqrt(1-(e*np.sin(phi))**2)

# This uses an equation for h (km) that does not diverge near the poles


def calc_h(phi, eci_vec):
    Rn = calc_Rn(phi)
    if abs(phi) > np.deg2rad(80):
        z = eci_vec[2]
        L = z + e**2 * Rn * np.sin(phi)
        h = (L/np.sin(phi)) - Rn
    else:
        p = np.sqrt(eci_vec[0]**2 + eci_vec[1]**2)
        h = (p / np.cos(phi)) - Rn
    return h

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

# Tangent point as a function of time. Within this function, we have to do another Newton's method to find the "n" index for which 
def f(t, s_unit, R_orbit):
    # distance of half los for in-plane crossing
    A_2d = np.sqrt(R_orbit ** 2 - b_e ** 2)
    # km, max distance along LOS to look for grazing, a 3d los is always shorter than a 2d

    # Make an inital guess of the tangent point, using vectorization
    n_list = np.arange(0, 1.1*A_2d, 0.1)
    n_column_vec = n_list.reshape((len(n_list), 1))
    starArray = np.ones((len(n_list), 3)) * s_unit
    los_array = r(t, R_orbit) + n_column_vec * starArray
    # List of magnitudes of poins along the LOS
    p_mag_list = np.linalg.norm(los_array, axis=1)
    A_3d=0.1*np.argmin(p_mag_list)

    # Initialize gradient descent and secant method
    learn_rate = 0.001
    n = A_3d   # initial guess for n
    n_last = A_3d + 1
    alt_tol = 1e-5  # km, 1cm error tolerance for tangent altitude
    alt_tp = 1000.0
    alt_tp_last = eci2llh(r(t, R_orbit) + n_last * s_unit)[2]
    num_iter = 0
    while(abs(alt_tp - alt_tp_last) > alt_tol and num_iter < 20):
        alt_tp = eci2llh(r(t, R_orbit) + n * s_unit)[2]
        dh_dn = (alt_tp - alt_tp_last)/(n-n_last)  # Gradient
        n_last = n.copy()
        alt_tp_last = alt_tp.copy()
        n -= learn_rate*dh_dn
        alt_tp -= learn_rate*dh_dn**2
        num_iter += 1

    return alt_tp, num_iter, n

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

    A_2d = np.sqrt(R_orbit ** 2 - b_e ** 2)
    r0_2d = b_e * g_unit_proj - A_2d * s_proj

    # list of errors from r0_2d
    dr = np.linalg.norm(r(t_orbit, R_orbit) - r0_2d, axis=1)
    t1_index = np.argmin(dr)
    t1 = t_orbit[t1_index]  # t_0,guess

    # Newton's method to minimize f(t)
    t = t1  # initial guess
    t_last = t1 - 1  # initial guess for secant method

    b_last = 2*R_orbit  # initialization to enter for loop
    delta = 1.0  # sec time error, initialization to enter for loop
    # 75 m, altitude tolerance for identifying the graze point (max difference between  geodetic and geocentric altitude)
    graze_tolerance = 1e-3  # km
    num_iter = 1
    while(abs(b_last) > graze_tolerance and num_iter < 25):
        b, num_iter_n, n = f(t, s_unit, R_orbit)
        m = (f(t, s_unit, R_orbit)[0] - f(t_last, s_unit, R_orbit)[0])/(t-t_last)
        if b is np.nan or m is np.nan:
            # No solution found (r0_hc will have a 'nan' in it)
            break
        print(f"num_iter = {num_iter_n} to get n = {n} km, alt_tp = {b} km")
        delta = b/m
        t -= delta
        b_last = b
        t_last = t
        print(f"--- FINISHED {num_iter} ITERATION IN TIME ---")
        num_iter += 1

    # If we broke out of the loop, r0_hc will include a 'nan'
    if b is np.nan or m is np.nan or num_iter >= 25:
        r0_hc = np.array([np.nan, np.nan, np.nan])
    else:
        r0_hc = r(t, R_orbit)
    print(f"psi = {psi_deg} deg")
    print(f"t0_model = {t} sec")
    print(f"{num_iter} iterations over time")
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

# This function calculates and plots r0_hc for a single orbit for psi
# Input: H is the orbital alitude above R_planet (km)
# Input: d_psi is the step size (deg) in out-of-plane angle

def main(R_orbit, psi):
    # find r0 for psi
    r0_hc, s_unit, num_iter = rotate_and_find_r0hc(psi, R_orbit)
    print(f"r0_hc = {r0_hc}")
    return 0


if __name__ == '__main__':
    # Consider an equatorial Earth orbit at H=420 km:
    main(R_orbit=a_e+420, psi=5)

    # r0_km = np.array([-4512.40+200, 3844.34-200, -3326.14+200])

    # # Code to test geocentric to geodetic algorithm
    # lat, lon, height = eci2llh(r0_km)

    # print(height)
    # print(np.linalg.norm(r0_km) - np.linalg.norm(point_on_earth_azimuth_polar(lon, (np.pi/2)-lat)))
