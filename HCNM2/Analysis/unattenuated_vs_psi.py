# Author: Nathaniel Ruhl
# This script examines the unattenuate count rate in a given energy band against the out-of-plane angle, psi

# Import local modules
from Modules.get_unattenuated_rate_RXTE import get_unattenuated_rate_RXTE

# Import observation dictionaries
from ObservationDictionaries.RXTE.all_dicts import all_dicts

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd

e_band_ch = [7-1, 9]

for obs_dict in all_dicts:
    obsid = obs_dict["obsID"]
    # string identifier for the given energy band
    e_id = f"{e_band_ch[0]+1}_{e_band_ch[1]}"

    scope_fn = f"Data/RXTE/{obs_dict['obsID']}/time_ra_dec.npy"
    trd = np.load(scope_fn)

    # Identify telescopic pointing in crossing time range
    indices = np.where((trd[:, 0] >= obs_dict['crossing_time_range'][0]) & (
        trd[:, 0] <= obs_dict['crossing_time_range'][1]))[0]

    ra_mean = np.mean(trd[indices, 1])
    dec_mean = np.mean(trd[indices, 2])

    # Determine unattenuated rate
    # Read in the data file

    fn_rateTime = cwd + f"/Data/RXTE/{obsid}/matrices/{e_id}_matrices/{e_id}_rateTime.npy"
    rateTime = np.load(fn_rateTime)
    rate_data_raw = rateTime[:, 0]
    time_data_raw = rateTime[:, 1]

    N0 = get_unattenuated_rate_RXTE(obs_dict, rate_data_raw, time_data_raw)

    plt.figure(1)
    plt.scatter(N0, ra_mean, label=obsid)

    plt.figure(2)
    plt.scatter(N0, dec_mean, label=obsid)

plt.figure(1)
plt.ylabel("RA")
plt.xlabel(r"$N_0$")
plt.legend()

plt.figure(2)
plt.ylabel("DEC")
plt.xlabel(r"$N_0$")
plt.legend()

plt.show()

