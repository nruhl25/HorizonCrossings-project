# Author: Nathaniel Ruhl

import numpy as np

# This script contains the ADM to predict y50 based on the SN and TOD of an observation
# Note that obs_dict should contain the fields that are used in the ADM before this function is called

def predict_y50(obs_dict, e_band_kev):
    # For now we will use the ADM for the lowest energy band for all the other energy bands, just so it is defined, but we should really have logic in here (example below) to use the correct ADM.

    # if all(e_band_kev == np.array([2.87, 4.09])):
    #     y50 = 30...
    # else:
    #     y50 = 40...
    #     # print("The ADM is not known for the specified e-band, so we are using the ADM for 2.87 keV - 4.09 keV as default")

    y50 = 6496.60578784075 - 0.007420604673815521 * \
        obs_dict['SN']-6.292346683662917*obs_dict['TOD']
    return y50
