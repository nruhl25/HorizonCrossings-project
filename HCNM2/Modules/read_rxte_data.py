# Author: Nathaniel Ruhl
# This script contains a function to pre-process RXTE data at run-time which is used in the "RXTE Driver"

from Modules.generate_nans_rxte import generate_nans_rxte
from Preprocessing.RXTE_channel_to_keV import channel_to_keV_epoch5
from Modules.get_unattenuated_rate_RXTE import get_unattenuated_rate_RXTE

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd

# This function reads in the RXTE data and returns the 4 arrays that Jacob created and unattenuated_rate and e_band_kev
def read_rxte_data(obs_dict, e_band_ch):
    e_band_kev = channel_to_keV_epoch5(e_band_ch)
    # vars below used for reading in the correct data files
    obsid = obs_dict["obsID"]
    # string identifier for the given energy band
    e_id = f"{e_band_ch[0]+1}_{e_band_ch[1]}"
    fn_rateTime = cwd + \
        f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_rateTime.npy"
    fn_ampCenters = cwd + \
        f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_ampCenters.npy"

    # 3) Read in the data files
    rateTime = np.load(fn_rateTime)
    ampCenters = np.load(fn_ampCenters)

    rate_data_raw = rateTime[:, 0]
    time_data_raw = rateTime[:, 1]
    normalized_amplitudes = ampCenters[:, 0]
    bin_centers_kev = ampCenters[:, 1]   # keV

    unattenuated_rate = get_unattenuated_rate_RXTE(
        obs_dict, rate_data_raw, time_data_raw)

    # 4) Lengthen rate_data and time_data if necessary
    rate_data, time_data = generate_nans_rxte(
        obs_dict, rate_data_raw, time_data_raw)
    return rate_data, time_data, normalized_amplitudes, bin_centers_kev, unattenuated_rate, e_band_kev
