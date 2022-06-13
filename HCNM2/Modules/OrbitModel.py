# Author: Nathaniel Ruhl
# This class contains methods to define an orbit model for the horizon crossing duration

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d


class OrbitModel:

    # INPUTS: obs_dict (dict), orbit_model = "mkf", "aster", or "circle"
    # OUTPUTS: position and time at intervals of time step
    @staticmethod
    def define_orbit_model(obs_dict, orbit_model, time_step):
        t0 = obs_dict["crossing_time_range"][0]
        tf = obs_dict["crossing_time_range"][1]
        t_array = np.arange(t0, tf, time_step)   # time array of high-definition HC positions
        if orbit_model == "mkf":
            r_mkf, v_mkf, t_mkf = OrbitModel.read_mkf(obs_dict["mkf_path"])
            r_array = OrbitModel.interpolate_mkf_state(r_mkf, t_mkf, t_array)
            v_array = OrbitModel.interpolate_mkf_state(v_mkf, t_mkf, t_array)
        # temporary solution until we implement read_aster_orbit()
        elif orbit_model == "aster":
            r_aster, v_aster, t_aster = OrbitModel.read_mkf(obs_dict["mkf_path"])
            r_array = OrbitModel.interpolate_mkf_state(r_aster, t_aster, t_array)
            v_array = OrbitModel.interpolate_mkf_state(v_aster, t_aster, t_array)
        return r_array, v_array, t_array

    # Reads the orbital state from NICER's mkf file
    @staticmethod
    def read_mkf(fn_string):
        tab_mkf = Table.read(fn_string, hdu=1)
        r = np.array(tab_mkf['POSITION'])
        t = np.array(tab_mkf['TIME'])
        v = np.array(tab_mkf['VELOCITY'])
        return r, v, t

    # time_array is the list of times that corresponds to r_array or v_array, the state "x_mkf"
    @staticmethod
    def interpolate_mkf_state(x_mkf, t_mkf, t_array):
        x_interpolator = interp1d(t_mkf, x_mkf[:, 0], kind="cubic")
        y_interpolator = interp1d(t_mkf, x_mkf[:, 1], kind="cubic")
        z_interpolator = interp1d(t_mkf, x_mkf[:, 2], kind="cubic")

        x_x = x_interpolator(t_array).reshape((len(t_array), 1))
        x_y = y_interpolator(t_array).reshape((len(t_array), 1))
        x_z = z_interpolator(t_array).reshape((len(t_array), 1))
        r_array = np.hstack((x_x, x_y, x_z))

        return r_array


