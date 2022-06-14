# Author: Nathaniel Ruhl

# This class defines the parameters necessary to describe the circular orbit of the horizon crossing, it is a subclass of Planet

import numpy as np

# import local modules
from Planet import Planet, G


class Orbit(Planet):
    def __init__(self, cb, H):
        Planet.__init__(self, cb)
        self.H = H   # km, orbital altitude
        self.R_orbit = self.R + self.H   # km, orbital radius
        self.T = self.radius_to_period()  # sec, orbital period
        self.omega = 2*np.pi/self.T    # rad/sec, angular velocity
        self.theta = np.arcsin(self.R/self.R_orbit)  # rad, angle characteristic angle of the orbit and central body

    def radius_to_period(self):
        R_m = self.R_orbit * 10 ** 3   # convert radius to meters
        T = np.sqrt((4 * np.pi ** 2 * R_m ** 3) /
                    (G * self.M))   # sec
        return T
    
    @property
    def epsilon_final(self):
        ef = (np.pi/2) - self.theta
        return ef

    # Functions for Kepler's Law
    # Functions for Kepler's Law, semi-major axis (a, R_orbit for a circle) and orbital period (T), using known mass of earth

    def period_to_a(self, T):
        a = (((T ** 2) * G * self.M /
            (4 * np.pi ** 2)) ** (1. / 3)) / (10 ** 3)  # km
        return a


    def a_to_period(self, a_km):
        a_m = a_km * 10 ** 3   # convert a to meters
        T = np.sqrt((4 * np.pi ** 2 * a_m ** 3) /
                    (G * self.M))   # sec
        return T

    @property
    def time_final(self):
        tf = self.epsilon_final/self.omega
        return tf


if __name__ == "__main__":
    ISS = Orbit(cb="Earth", H=420)
    print(ISS.time_final)
