# Author: Nathaniel Ruhl

import numpy as np

# import modules from HCNM_Analysis/
import sys
sys.path.append("/Users/nathanielruhl/Desktop/HorizonCrossings-Summer22/nruhl_final_project/HCNM_Analysis/")
import tools as tools


# This class contains info for the orbservation and corresponding circular orbit
class ObservationOrbit:
    t_step_size = 0.01
    def __init__(self, obs_dict_raw, hc_type):
        # unnpack input dictionay
        # Note that i, raan, aop correspond to the actual oservation type,
        # not the desired hc_type, so we won't make attributes here
        # (we could later define them based on h_unit, ref. Bate text eqns)
        self.T = obs_dict_raw["T"]
        self.RA_SOURCE = obs_dict_raw["RA_SOURCE"]
        self.DEC_SOURCE = obs_dict_raw["DEC_SOURCE"]
        self.hc_type = hc_type

        # Define derived fields
        self.R_orbit = tools.period_to_a(self.T) # sec
        self.OMEGA_ORB = 2*np.pi/self.T  # rad/sec
        self.starECI = tools.celestial_to_geocentric(self.RA_SOURCE, self.DEC_SOURCE)
        if hc_type == "rising":
            self.h_unit = tools.get_h_unit(obs_dict_raw["i"], obs_dict_raw["raan"], obs_dict_raw["aop"])
        elif hc_type == "setting":
            self.h_unit = -tools.get_h_unit(obs_dict_raw["i"], obs_dict_raw["raan"], obs_dict_raw["aop"])
        self.starECI_proj = tools.proj_on_orbit(self.starECI, self.h_unit)
        self.Q = tools.get_Q_matrix(obs_dict_raw["i"], obs_dict_raw["raan"], obs_dict_raw["aop"], hc_type)  # converts from ECI to perifocal
        self.T = tools.a_to_period(self.R_orbit)   # orbital period
        self.t_orbit = np.arange(0, int(self.T), ObservationOrbit.t_step_size)
        self.orbit_model = self.get_orbit_vec_circle()

    def get_orbit_vec_circle(self):
        # Define the orbital circle in the perifocal frame, then transform to ECI

        # Make use of python broadcasting to fill the perifocal array
        t_orbit_column = self.t_orbit.reshape((len(self.t_orbit), 1))
        x_per = self.R_orbit * np.cos(self.OMEGA_ORB * t_orbit_column)
        y_per = self.R_orbit * np.sin(self.OMEGA_ORB * t_orbit_column)
        z_per = np.zeros(t_orbit_column.shape)
        orbit_vec_per = np.hstack((x_per, y_per, z_per))

        orbit_vec_eci = np.zeros(orbit_vec_per.shape)

        # I don't know how to do this operation with broadcasting
        for ti, time in enumerate(self.t_orbit):
            orbit_vec_eci[ti] = np.dot(np.matrix.transpose(self.Q), orbit_vec_per[ti])

        return orbit_vec_eci
