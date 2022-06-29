# Author: Nathaniel Ruhl
# The function sourceIdentifier(i, raan, H, source_ra, source_dec) takes in the inclination/raan/altitude of an orbit and source position in ra/dec (deg), and says whether source is a potential HC candidate or not

R_earth = 6371.0  # km

import numpy as np
import matplotlib.pyplot as plt

from psi_solver import find_r0hc

# Transforms RA/DEC into an ECI unit vector
def celestial_to_geocentric(alpha, delta):
    x = np.cos(delta)*np.cos(alpha)
    y = np.cos(delta)*np.sin(alpha)
    z = np.sin(delta)
    return np.array([x, y, z])

def geocentric_to_celestial(unit_vector):
    delta = np.arcsin(unit_vector[2])
    alpha = np.arctan2(unit_vector[1], unit_vector[0])
    return alpha, delta

# This function takes in a lon unit vector and returns the raan
def get_raan(n_unit):
    if n_unit[1] < 0:
        raan = 2 * np.pi - np.arccos(n_unit[0])
    else:
        raan = np.arccos(n_unit[0])
    return raan

# This function returns the pole vector in ECI coordinates (third row of Q)
def get_h_unit(i, raan):
    h_eci = np.array([np.sin(raan) * np.sin(i),
                      -np.cos(raan) * np.sin(i),
                      np.cos(i)])
    return h_eci

# Everything is in ECI coordinates
def get_n_unit(h_unit):
    n_unit = np.cross(np.array([0, 0, 1]), h_unit) / np.linalg.norm(np.cross(np.array([0, 0, 1]), h_unit))
    return n_unit

def theta_max(H, R_planet=R_earth):
    theta = np.arctan2(R_planet, np.sqrt((R_planet+H)**2-R_planet**2))
    return theta

# Transform a vector v_e into v_p (n_unit and h_unit are the 1 and 3 unit vectors)
# This is not currently being used
def eci2perifocal(n_unit, h_unit, v_e):
    T = np.column_stack((n_unit, np.cross(h_unit, n_unit), h_unit)).T
    v_p = np.dot(T, v_e)
    return v_p

def perifocal2eci(n_unit, h_unit, v_p):
    T = np.column_stack((n_unit, np.cross(h_unit, n_unit), h_unit))
    v_e = np.dot(T, v_p)
    return v_e

def generate_i_raan():
    raan = 360 * np.random.ranf()
    i = 360 * np.random.ranf() - 180
    return i, raan

# This is the main function identify a source as a horizon crossing candidate
# H: [km]
# i, raan, source_ra, source_dec: [deg]
def sourceIdentifier(i, raan, H, source_ra, source_dec):
    s_unit = celestial_to_geocentric(np.deg2rad(source_ra), np.deg2rad(source_dec))  # ECI
    h_unit = get_h_unit(np.deg2rad(i), np.deg2rad(raan))   # ECI

    psi = (np.pi/2) - np.arccos(np.dot(h_unit, s_unit))  # out-of-plane angle to source

    if psi < theta_max(H):
        return True
    else:
        return False

# This function generates a source that is compatible the inputted orbit (i, raan, H)
# Return ra and dec of source in degrees
def generateSource(i, raan, H):
    # Construct random/compatible out-of-plane angle psi
    max_psi = theta_max(H)-np.deg2rad(1) # give a little breathing room for the locate r0 algorithm

    psi_random = 2*(max_psi)*np.random.ranf() - max_psi  # rad (only positive here!)
    alpha_random = (np.pi/2) - psi_random  # complementary angle mad with pole vector, rad
    h_comp = np.cos(alpha_random)  # component of s_unit along orbital pole
    p_comp = np.sqrt(1-h_comp**2)  # component of s_unit on orbital plane

    # Break down p_comp into a and b components
    nu_star = 2*np.pi*np.random.ranf()   # random source anomaly around orbital plane
    a_rand = np.sqrt(1-h_comp**2)*np.sin(nu_star)
    b_rand = np.sqrt(p_comp**2-a_rand**2)
    s_unit = np.array([a_rand, b_rand, h_comp])

    # transform into ECI frame
    h_unit = get_h_unit(np.deg2rad(i), np.deg2rad(raan))   # pole vector in ECI
    n_unit = get_n_unit(h_unit)   # line of nodes vector in ECI
    starECI = perifocal2eci(n_unit, h_unit, v_p=s_unit)

    ra, dec = geocentric_to_celestial(starECI)   # rad, transform to celestial coordinates

    return np.rad2deg(ra), np.rad2deg(dec), psi_random

def main():
    # Show that everything works with two examples:
    # print("V4641 Sgr. Crossing on feb3 2020: ")
    # print(sourceIdentifier(H=424.27, i=51.538, raan=293.1, source_ra=274.839, source_dec=-25.407))

    # print("NICER Crab Crossing: ")
    # print(sourceIdentifier(H=436.72, i=51.736, raan=67.99, source_ra=83.633, source_dec=22.014))

    oe_test = (50, 30, 420)   # i (deg), raan (deg), H (km). Test orbital elements
    R_orbit = R_earth + 420
    num_trials = 1000
    num_fails = 0
    for _ in range(num_trials):
        ra_test, dec_test, psi_test = generateSource(*oe_test)   # example compatible source
        s_unit = celestial_to_geocentric(np.deg2rad(ra_test), np.deg2rad(dec_test))  # ECI

        # Only do the LocateR0hc algorithm for a source that is going to work
        if sourceIdentifier(*oe_test, ra_test, dec_test) is False:
            print("Houston, we have a problem")
        else:
            r0_hc, num_iter = find_r0hc(s_unit, R_orbit)  # need to make this a function of h_unit(i, raan) in order to test num_fails correctly, maybe ???
            print(r0_hc)
            if any(np.isnan(r0_hc)):
                num_fails += 1

    print(f"{num_fails} fails out of {num_trials} trials")
    return 0


if __name__ == '__main__':
    main()
