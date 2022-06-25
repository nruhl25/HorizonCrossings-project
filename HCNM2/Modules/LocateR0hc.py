# Author: Nathaniel Ruhl
#  This script contains a class that locates r0_hc for an arbitrary orbital model
# For the algorithm to work, we must a-priori know hc_type

import numpy as np

import Modules.tools as tools
import Modules.constants as constants


# This class locates r0 for both the rising and setting crossing
class LocateR0hc:
    n_step_size = 0.1   # km step size along the line-of-sight (LOS)
    max_los_dist = 3000   # km, max distance that we look for graze point along the LOS

    def __init__(self, obs_dict, r_array, v_array, t_array):
        # Unpack inputs
        self.obs_dict = obs_dict
        self.hc_type = obs_dict["hc_type"]
        self.starECI = obs_dict["starECI"]
        self.crossing_time_range = obs_dict["crossing_time_range"]
        if obs_dict["detector"] == "NICER":
            self.year0 = 2014
        elif obs_dict["detector"] == "RXTE":
            self.year0 = 1994
        self.r_array = r_array
        self.v_array = v_array
        self.t_array = t_array

        # derived inputs
        self.R_orbit, self.h_unit = self.define_R_orbit_h_unit()
        self.psi = np.rad2deg(
            (np.pi/2)-np.arccos(np.dot(self.h_unit, self.starECI)))  # out-of plane angle (deg)


        # Sequential steps of the algorithm
        self.A_2d, self.r0_2d = self.get_initial_guess()
        self.t0_guess_list, self.r0_guess_list = self.get_t0_guess_indices()
        self.r0_hc, self.t0_model_index, self.graze_point, self.A_3d = self.locate_r0_numerical()

        # Other useful variables
        self.g_unit = self.graze_point / np.linalg.norm(self.graze_point)
        self.t0_model = t_array[self.t0_model_index]
        self.lat_gp, self.lon_gp, _ = tools.eci2geodetic_pymap_vector(self.graze_point, self.t0_model, self.year0)

    def get_initial_guess(self):
        starECI_proj = tools.proj_on_orbit(self.starECI, self.h_unit)
        # Use the 2d formulas to guess where r0 may be
        if self.hc_type == "rising":
            g_unit_proj = np.cross(starECI_proj, self.h_unit)
        elif self.hc_type == "setting":
            g_unit_proj = np.cross(self.h_unit, starECI_proj)

        A_2d = np.sqrt(self.R_orbit ** 2 - constants.R_EARTH ** 2)
        r0_2d = constants.R_EARTH * g_unit_proj - A_2d * starECI_proj
        return A_2d, r0_2d

    def get_t0_guess_indices(self):
        if self.obs_dict['detector'] == "RXTE":
            print(f"psi = {self.psi}")
            if abs(self.psi) < 5:
                search_factor = 0.005
            elif abs(self.psi) < 10:
                search_factor = 0.001
            elif abs(self.psi) < 20:
                search_factor = 0.05
            elif abs(self.psi) < 30:
                search_factor = 0.1
            else:
                search_factor = 0.1
                print("Out-of plane angle greater than 30 deg. search_factor = 0.1")
        else:
            # NICER doesn't get very far out-of-plane
            search_factor = 0.005
        # TODO: Improve this algorithm... make a sort of gradient descent to find the tangent point
        r0_guess_indices = np.isclose(self.r_array, self.r0_2d, search_factor)

        # 0.5% corresponds to ~15km or more for each component (0.005*3000=15)

        t0_guess_list = []  # INDICES in r_array

        for index, value in enumerate(r0_guess_indices):
            if all(value) == True:
                t0_guess_list.append(index)
        # get the positions that corresponds to the t0 list
        # t0 indices are for r_array
        r0_guess_list = self.r_array[min(t0_guess_list):max(t0_guess_list)+1]

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
        # print(constants.R_EARTH)
        for time_index, t0_model_index in enumerate(self.t0_guess_list):
            # Lists to check radial altitude at different points along the LOS
            n_list = np.arange(0, LocateR0hc.max_los_dist, LocateR0hc.n_step_size)
            los_points = self.los_line(time_index, n_list)  # all points along the LOS

            # Lists to check radial altitude at different points along the LOS
            # Pymap3d (tools.py) seems to be buggy right now, so we'll ignore longitude and do it manually
            los_mag_list = np.sqrt(los_points[:, 0] ** 2 + los_points[:, 1] ** 2 + los_points[:, 2] ** 2)
            polar_angles = np.arccos(los_points[:, 2] / los_mag_list)  # polar angle at every point along the line of sight
            # Find the radius of earth with the same polar angle as points along the line of sight
            earth_points = tools.point_on_earth_azimuth_polar(np.zeros_like(polar_angles), polar_angles)
            earth_radius_list = np.sqrt(earth_points[:, 0] ** 2 + earth_points[:, 1] ** 2 + earth_points[:, 2] ** 2)

            # Identify hc_type (note that this needs to be defined earlier)
            # if time_index == 0:
            #     middle_index_los = np.argmin(los_mag_list)
            #
            #     if los_mag_list[middle_index_los] < earth_radius_list[middle_index_los]:
            #         hc_type = "rising"
            #     elif los_mag_list[middle_index_los] > earth_radius_list[middle_index_los]:
            #         hc_type = "setting"

            # Check if we reached the tangent grazing point
            # print(np.min(los_mag_list)-np.min(earth_radius_list))
            if self.hc_type == "rising":
                if all(los_mag_list >= earth_radius_list):
                    # Find the point of closest approach, the tangent point
                    n_graze_index = np.argmin(los_mag_list)
                    A_3d = n_list[n_graze_index]
                    # The 2 below definitions are insightful, but not currently being used
                    graze_point = los_points[n_graze_index]
                    graze_phi = polar_angles[n_graze_index]   # polar angle at graze_point
                    return self.r0_guess_list[time_index], t0_model_index, graze_point, A_3d
                else:
                    continue
                    # keep moving through time until the whole LOS is above earth
            elif self.hc_type == "setting":
                if any(los_mag_list <= earth_radius_list):
                    # Find the point of closest approach, the tangent point
                    n_graze_index = np.argmin(los_mag_list)
                    A_3d = n_list[n_graze_index]
                    # The 2 below definitions are insightful, but not currently being used
                    graze_point = los_points[n_graze_index]
                    graze_phi = polar_angles[n_graze_index]   # polar angle at graze_point
                    return self.r0_guess_list[time_index], t0_model_index, graze_point, A_3d
                else:
                    # keep moving through time until the whole LOS is above earth
                    continue

        print('Tangent point not located in specified time range')
        return 0, 0, 0, 0

    # Used in HCNM Driver
    def return_r0_data(self):
        return self.t0_model_index, self.lat_gp, self.lon_gp

    # Function used to define R_orbit and h_unit at the middle of the crossing time period
    def define_R_orbit_h_unit(self):
        mid_time = (self.crossing_time_range[0]+self.crossing_time_range[1])/2
        mid_time_index = np.where(self.t_array >= mid_time)[0][0]
        R_orbit = np.linalg.norm(self.r_array[mid_time_index])
        h_unit = np.cross(self.r_array[mid_time_index], self.v_array[mid_time_index])
        h_unit = h_unit / np.linalg.norm(h_unit)
        return R_orbit, h_unit
