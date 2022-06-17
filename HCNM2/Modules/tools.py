# Author: Nathaniel Ruhl
# This library contains various tools for the analysis

import Modules.constants as constants

import math as m
import numpy as np
import datetime
import pymap3d as pm
import numbers

# Optimize pymap3d and astropy operations
from astropy.utils import iers   # cannot go before 1962
iers.conf.auto_download = False

# This function converts seconds from Jan 1 2014 into a datetime in UTC
def convert_time_NICER(time):
    timezero = datetime.datetime(year=2014, month=1,
                                 day=1, hour=0, minute=0, second=0)
    if isinstance(time, numbers.Real):
        new_time = timezero+datetime.timedelta(seconds=time)
        return new_time
    elif isinstance(time, list) or isinstance(time, np.ndarray):
        new_time_list = []
        for index, t in enumerate(time):
            new_time = timezero + datetime.timedelta(seconds=t)
            new_time_list.append(new_time)
        return np.array(new_time_list)
    else:
        raise RuntimeError('time must be MET number or array/list of times')


# This function converts seconds from Jan 1 1994 into a datetime in UTC
def convert_time_RXTE(time):
    timezero = datetime.datetime(year=1994, month=1,
                                 day=1, hour=0, minute=0, second=0)
    if isinstance(time, numbers.Real):
        new_time = timezero + datetime.timedelta(seconds=time)
        return new_time
    elif isinstance(time, list) or isinstance(time, np.ndarray):
        new_time_list = []
        for index, t in enumerate(time):
            new_time = timezero + datetime.timedelta(seconds=t)
            new_time_list.append(new_time)
        return np.array(new_time_list)
    else:
        raise RuntimeError('time must be MET number or array/list of times')

# This function does the above 2 conversions together
def convert_time(time, year0):
    timezero = datetime.datetime(year=year0, month=1,
                                 day=1, hour=0, minute=0, second=0)
    if isinstance(time, numbers.Real):
        new_time = timezero + datetime.timedelta(seconds=time)
        return new_time
    elif isinstance(time, list) or isinstance(time, np.ndarray):
        new_time_list = []
        for index, t in enumerate(time):
            new_time = timezero + datetime.timedelta(seconds=t)
            new_time_list.append(new_time)
        return np.array(new_time_list)
    else:
        raise RuntimeError('time must be MET number or array/list of times')


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


# theta is azimuthal angle, phi is polar (weird, right?)
def point_on_earth_azimuth_polar(theta_list, phi_list):
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


# The functions below take in eci points (in km) and return latitude, longitude, and altitude using pymap3d
#  note that time seconds is a single time, float number

# multiple los_points, time_seconds is either a number or array
def eci2geodetic_pymap_array(los_point_array, time_seconds, year0):
        los_point_array_m = los_point_array * 1000  # convert to [m]
        lats, lons, alts = pm.eci2geodetic(los_point_array_m[:, 0], los_point_array_m[:, 1], los_point_array_m[:, 2], convert_time(time_seconds, year0))
        return lats, lons, alts / 1000


# single los point to be converted
def eci2geodetic_pymap_vector(los_point_km, time_seconds, year0):
    los_point_m = los_point_km * 1000  # convert to [m]
    lat, lon, alt = pm.eci2geodetic(los_point_m[0], los_point_m[1], los_point_m[2], convert_time(time_seconds, year0))
    # return altitude in km, lat and lon in degrees
    return lat, lon, alt / 1000


