# This file defines the average ra/dec of RXTE during the time of the horizon crossing

# Import observation dictionaries
from ObservationDictionaries.RXTE.all_dicts import *

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd

# This function takes in an obs_dict and returns the average ra and dec during the crossing
def get_ra_dec(obs_dict):
    obsid = obs_dict["obsID"]
    arr = np.load(f"Data/RXTE/{obsid}/time_ra_dec.npy")
    t = arr[:,0]
    ra = arr[:,1]
    dec = arr[:,2]
    time_range_crossing = obs_dict["crossing_time_range"]

    # focus in to the crossing time range
    # Will get 3 or 4 data points in the time range
    crossing_indices = np.where((t >= time_range_crossing[0]) & (
        t <= time_range_crossing[1]))[0]
    ra_short = ra[crossing_indices]
    dec_short = dec[crossing_indices]

    return np.mean(ra_short), np.mean(dec_short)

def main():
    all_dicts = [dict_91802, dict_60079, dict_50099, dict_40805, dict_50098]

    for obs_dict in all_dicts:
        print(f"obsID = {obs_dict['obsID']}")
        ra, dec = get_ra_dec(obs_dict)
        print(f"Average ra = {ra} deg")
        print(f"Average dec = {dec} deg")
        print("-----------------")

    return 0

if __name__=="__main__":
    main()
