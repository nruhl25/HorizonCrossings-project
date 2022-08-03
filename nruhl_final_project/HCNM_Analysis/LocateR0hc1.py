# Author: Nathaniel Ruhl
# This script tests the method of Locating r0 for both a rising and setting crossing. In this script, we will use the orbital parameters corresponding to the V4641 Sgr. Horizon Crossing

# import standard libraries
import numpy as np
import matplotlib.pyplot as plt

# import modules from HCNM_Analysis/
import sys
sys.path.append("/Users/nathanielruhl/Desktop/HorizonCrossings-Summer22/nruhl_final_project/HCNM_Analysis/")
from ObservationOrbit import ObservationOrbit # Subclass of LocateR0hc2
import tools as tools
import constants as constants

# This class locates r0 for both the rising and setting crossing
class LocateR0hc2(ObservationOrbit):
    n_step_size = 0.1   # km step size along the line-of-sight (LOS)
    max_los_dist = 4000   # km, max distance that we look for graze point along the LOS
    def __init__(self, obs_dict_raw, hc_type):
        ObservationOrbit.__init__(self, obs_dict_raw, hc_type)
        self.A_2d, self.r0_2d = self.get_initial_guess()
        self.t0_guess_list, self.r0_guess_list = self.get_t0_guess_indices()
        self.r0_hc, self.g_unit, self.A_3d, self.t0_model = self.locate_r0_numerical()
        # g_unit can be used to define lat/lon of grazing point

    def get_initial_guess(self):
        # Use the 2d formulas to guess where r0 may be
        if self.hc_type == "rising":
            g_unit_proj = np.cross(self.starECI_proj, self.h_unit)
        elif self.hc_type == "setting":
            g_unit_proj = np.cross(self.h_unit, self.starECI_proj)

        A_2d = np.sqrt(self.R_orbit ** 2 - constants.R_EARTH ** 2)
        r0_2d = constants.R_EARTH * g_unit_proj - A_2d * self.starECI_proj
        return A_2d, r0_2d

    def get_t0_guess_indices(self):
        r0_guess_indices = np.isclose(self.orbit_model, self.r0_2d, 0.005)

        # 0.5% corresponds to ~15km or more for each component (0.005*3000=15)

        t0_guess_list = []  # indices in orbit_vec

        for index, value in enumerate(r0_guess_indices):
            if all(value) == True:
                t0_guess_list.append(index)

        # get the positions that corresponds to the t0 list
        # t0 indices are for orbit_vec
        r0_guess_list = self.orbit_model[min(t0_guess_list):max(t0_guess_list)+1]

        return t0_guess_list, r0_guess_list

    # Line of sight from the predicted satellite position r(t)
    def los_line(self, time_index, n_list):
        if isinstance(n_list, int) or isinstance(n_list, float):
            # n_list is not a list, but a single number
            n = n_list
            return self.r0_guess_list[time_index] + n * self.starECI
        else:
            n_column_vec = n_list.reshape((len(n_list), 1))
            starArray = np.ones((len(n_list), 3)) * self.starECI
            return self.r0_guess_list[time_index] + n_column_vec * starArray

    # Locate r0 via aligning the LOS to be tangent to earth
    def locate_r0_numerical(self):
        # Loop through different times, different lines of sight during the crossing
        for time_index, time in enumerate(self.t0_guess_list):
            # Lists to check radial altitude at different points along the LOS
            n_list = np.arange(0, LocateR0hc2.max_los_dist, LocateR0hc2.n_step_size)
            los_points = self.los_line(time_index, n_list)  # all points along the LOS

            # Lists to check radial altitude at different points along the LOS
            # Variable below is distance from point to origin, not los length
            los_mag_list = np.sqrt(los_points[:, 0] ** 2 + los_points[:, 1] ** 2 + los_points[:, 2] ** 2)
            phi_list = np.arccos(los_points[:, 2] / los_mag_list)
            # polar angle at every point along the line of sight
            # Find the radius of earth with the same polar angle as points along the line of sight
            earth_points = tools.point_on_earth(np.zeros_like(phi_list), phi_list)
            earth_radius_list = np.sqrt(earth_points[:, 0] ** 2 + earth_points[:, 1] ** 2 + earth_points[:, 2] ** 2)

            # Identify self.hc_type
            # This is a good method, but the problem is that it needs to be defined earlier
            # in a previous function
            if time_index == 0:
                middle_index_los = np.argmin(los_mag_list)
                if los_mag_list[middle_index_los]<earth_radius_list[middle_index_los]:
                    print("rising")
                elif los_mag_list[middle_index_los]>earth_radius_list[middle_index_los]:
                    print("setting")

            # Check if we reached the tangent grazing point, different for two hc types
            # This will be achieved for multiple lines of sight
            if self.hc_type == "rising":
                if all(los_mag_list >= earth_radius_list):
                    # Find the point of closest approach, the tangent point
                    n_graze_index = np.argmin(los_mag_list)
                    A_3d = n_list[n_graze_index]
                    # The 2 below definitions are insightful, but not currently being used
                    graze_point = los_points[n_graze_index]
                    graze_phi = phi_list[n_graze_index]   # polar angle at graze_point
                    t0_model = time_index / LocateR0hc2.t_step_size
                    return self.r0_guess_list[time_index], graze_point, A_3d, t0_model
                else:
                    # keep moving through time until the whole LOS is above earth
                    continue
            elif self.hc_type == "setting":
                if any(los_mag_list <= earth_radius_list):
                    # Find the point of closest approach, the tangent point
                    n_graze_index = np.argmin(los_mag_list)
                    A_3d = n_list[n_graze_index]
                    # The 2 below definitions are insightful, but not currently being used
                    graze_point = los_points[n_graze_index]
                    graze_phi = phi_list[n_graze_index]   # polar angle at graze_point
                    t0_model = time_index / LocateR0hc2.t_step_size
                    return self.r0_guess_list[time_index], graze_point, A_3d, t0_model
                else:
                    # keep moving through time until the whole LOS is above earth
                    continue

        print('Tangent point not located in specified time range')
        return 0, 0, 0, 0


# Observation dictionary
v4641 = {"i": np.deg2rad(51.538), # rad
        "raan": np.deg2rad(293.100), # rad
        "aop": 0.0, # rad
        "T": 5576.7, # sec
        "RA_SOURCE": np.deg2rad(274.839), # rad
        "DEC_SOURCE": np.deg2rad(-25.407)}

if __name__ == '__main__':
    obs = LocateR0hc2(v4641, "rising")
    print(obs.R_orbit)
    print(obs.t0_model)
    print(obs.r0_2d)
    print(obs.r0_hc)
    obs2 = LocateR0hc2(v4641, "setting")
    print(obs2.t0_model)
    print(obs2.r0_2d)
    print(obs2.r0_hc)

    # Confirmation that they are the same orbit
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(obs.orbit_model[:,0], obs.orbit_model[:,1], obs.orbit_model[:,2], label="rising")
    ax.plot3D(obs2.orbit_model[:,0], obs2.orbit_model[:,1], obs2.orbit_model[:,2],
    label="setting")
    plt.legend()
    plt.show()
