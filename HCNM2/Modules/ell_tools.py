# Author: Nathaniel Ruhl
# This script contains functions to convert from an eci vector to geodetic latitude and ellipsoidal altitude. It uses conversions from (Clynch, 2008)

import numpy as np

from Modules import constants

# This function converts an eci coordinate to lat/lon/height using the geocentric to geodetic altitude conversion (Clynch, 2008)
# INPUTS: eci_vec (km)
# OUTPUTS: lat (rad), lon (rad) (not from greenwhich), height (km)


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
    h = calc_h(phi_now, eci_vec)
    h_last = 1000000.0  # initialize to enter for loop
    tolerance = 1e-5   # [km] --> 10 cm
    # print(f"h_init = {h}")
    while abs(h - h_last) > tolerance:
        Rn = calc_Rn(phi_now)
        phi_next = np.arctan((z/p)*(1-(Rn/(Rn+h))*constants.e**2)**(-1.))
        h_last = h
        h = calc_h(phi_next, eci_vec)
        phi_now = phi_next

    lat = phi_now
    # print(f"h={h} km, lat = {np.rad2deg(lat)} deg")

    return lat, lon, h

# Helper functions for geocentric to geodetic conversion
# INPUT: geodetic lat (rad)
# OUTPUT: Rn (km)


def calc_Rn(phi):
    return constants.a / np.sqrt(1-(constants.e*np.sin(phi))**2)

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
