# Author: Nathaniel Ruhl
# This library contains various tools for the analysis

import math as m
import numpy as np

import sys
sys.path.append("/Users/nathanielruhl/Desktop/HorizonCrossings-Summer22/nruhl_final_project/HCNM_Analysis/")
import constants as constants

# This matrix transforms an ECI vector into the perifocal frame {p,q,h}
def get_Q_matrix(i, raan, aop, hc_type):
    p_eci = np.array([m.cos(raan) * m.cos(aop) - m.sin(raan) * m.sin(aop) * m.cos(i),
    m.sin(raan) * m.cos(aop) + m.cos(raan) * m.cos(i) * m.sin(aop),
    m.sin(i) * m.sin(aop)])

    q_eci = np.array([-m.cos(raan) * m.sin(aop) - m.sin(raan) * m.cos(i) * m.cos(aop),
    -m.sin(raan) * m.sin(aop) + m.cos(raan) * m.cos(i) * m.cos(aop),
    m.sin(i) * m.cos(aop)])

    h_eci = np.array([m.sin(raan) * m.sin(i),
    -m.cos(raan) * m.sin(i),
    m.cos(i)])

    if hc_type == "rising":
        Q = np.array([list(p_eci),
                    list(q_eci),
                    list(h_eci)])
    elif hc_type == "setting":
        Q = np.array([list(q_eci),
                    list(p_eci),
                    list(-h_eci)])
    return Q

# This function returns the pole vector in eci coordinates
def get_h_unit(i, raan, aop):
    h_eci = np.array([m.sin(raan) * m.sin(i),
    -m.cos(raan) * m.sin(i),
    m.cos(i)])
    return h_eci

# Transforms RA/DEC into an ECI unit vector
def celestial_to_geocentric(alpha, delta):
    x = np.cos(delta)*np.cos(alpha)
    y = np.cos(delta)*np.sin(alpha)
    z = np.sin(delta)
    return np.array([x, y, z])


def geocentric_to_celestial(unit_vector):
    delta = np.arcsin(unit_vector[2])
    alpha = np.arcsin(unit_vector[1]/np.cos(delta))
    return alpha, delta


# Function to project the source onto the plane of the orbit
def proj_on_orbit(r_source, h_unit):
    r_prime_source = r_source - h_unit * np.dot(r_source, h_unit)   # project on orbit plane
    r_prime_source = r_prime_source / np.linalg.norm(r_prime_source)  # normalize the vector
    return r_prime_source


# This function uses Kepler's 3rd law to convert semi-major axis into
def a_to_period(a_km):
    a_m = a_km * 10 ** 3   # convert a to meters
    T = np.sqrt((4 * np.pi ** 2 * a_m ** 3) / (constants.G * constants.M_EARTH))   # sec
    return T

def period_to_a(T):
    a = (((T ** 2) * constants.G * constants.M_EARTH / (4 * np.pi ** 2)) ** (1. / 3)) / (10 ** 3)  # km
    return a

def point_on_earth(theta_list, phi_list):
    if isinstance(phi_list, np.integer) or isinstance(phi_list, np.float):
        # theta and phi are single values
        phi = phi_list
        theta = theta_list
        x = constants.a * np.cos(theta) * np.sin(phi)
        y = constants.b * np.sin(theta) * np.sin(phi)
        z = constants.c * np.cos(phi)
        return np.array([x, y, z])
    else:
        # theta and phi are BOTH lists
        phi_column_vec = phi_list.reshape((len(phi_list),1))
        theta_column_vec = theta_list.reshape((len(theta_list),1))
        x = constants.a * np.cos(theta_column_vec) * np.sin(phi_column_vec)
        y = constants.b * np.sin(theta_column_vec) * np.sin(phi_column_vec)
        z = constants.c * np.cos(phi_column_vec)
        return np.hstack((x, y, z))

print(period_to_a(5576.7))
