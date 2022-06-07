# Author: Nathaniel Ruhl
#  This script contains a class that locates r0_hc for an arbitrary orbital model
# For the algorithm to work, we must a-priori know hc_type

import numpy as np

import tools as tools
import constants as constants

# This class locates r0 for both the rising and setting crossing
class LocateR0hc:
    n_step_size = 0.1   # km step size along the line-of-sight (LOS)
    max_los_dist = 4000   # km, max distance that we look for graze point along the LOS
    def __init__(self, obs_dict, r_array, t_array):
        # Unpack inputs
        self.hc_type = obs_dict["hc_type"]
        self.starECI = obs_dict["starECI"]
        self.h_unit = obs_dict['h_unit']
        self.R_orbit = obs_dict['R_orbit']
        self.r_array = r_array
        self.t_array = t_array

        # Sequential steps of the algorithm
        self.A_2d, self.r0_2d = self.get_initial_guess()
        self.t0_guess_list, self.r0_guess_list = self.get_t0_guess_indices()
        self.r0_hc, self.t0_model_index, self.graze_point, self.A_3d = self.locate_r0_numerical()

        # Other useful variables
        self.g_unit = self.graze_point / np.linalg.norm(self.graze_point)
        self.t0_model = t_array[self.t0_model_index]

    # The method below is called in HCNM_Driver
    def return_r0_hc_params(self):
        # Calculate latitude and longitude of the graze point
        lat_gp, lon_gp, alt_gp_meters = 15, 15, 0.2 # tools.eci2geodetic_pymap(self.graze_point, self.t0_model)
        # pymap3d is not currently working
        return self.t0_model_index, lat_gp, lon_gp

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
        r0_guess_indices = np.isclose(self.r_array, self.r0_2d, 0.005)

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
        for time_index, t0_model_index in enumerate(self.t0_guess_list):
            # Lists to check radial altitude at different points along the LOS
            n_list = np.arange(0, LocateR0hc.max_los_dist, LocateR0hc.n_step_size)
            los_points = self.los_line(time_index, n_list)  # all points along the LOS

            # Lists to check radial altitude at different points along the LOS
            # Variable below is distance from point to origin, not los length
            los_mag_list = np.sqrt(los_points[:, 0] ** 2 + los_points[:, 1] ** 2 + los_points[:, 2] ** 2)
            phi_list = np.arccos(los_points[:, 2] / los_mag_list)
            # polar angle at every point along the line of sight
            # Find the radius of earth with the same polar angle as points along the line of sight
            earth_points = tools.point_on_earth(np.zeros_like(phi_list), phi_list)
            earth_radius_list = np.sqrt(earth_points[:, 0] ** 2 + earth_points[:, 1] ** 2 + earth_points[:, 2] ** 2)
            # Should I use pymap3d altitudes?

            # Identify hc_type (note that this needs to be defined earlier)
            if time_index == 0:
                middle_index_los = np.argmin(los_mag_list)
                if los_mag_list[middle_index_los]<earth_radius_list[middle_index_los]:
                    hc_type = "rising"
                elif los_mag_list[middle_index_los]>earth_radius_list[middle_index_los]:
                    hc_type = "setting"

            # Check if we reached the tangent grazing point
            if self.hc_type == "rising":
                if all(los_mag_list >= earth_radius_list):
                    # Find the point of closest approach, the tangent point
                    n_graze_index = np.argmin(los_mag_list)
                    A_3d = n_list[n_graze_index]
                    # The 2 below definitions are insightful, but not currently being used
                    graze_point = los_points[n_graze_index]
                    graze_phi = phi_list[n_graze_index]   # polar angle at graze_point
                    return self.r0_guess_list[time_index], t0_model_index, graze_point, A_3d
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
                    return self.r0_guess_list[time_index], t0_model_index, graze_point, A_3d
                else:
                    # keep moving through time until the whole LOS is above earth
                    continue

        print('Tangent point not located in specified time range')
        return 0, 0, 0, 0
