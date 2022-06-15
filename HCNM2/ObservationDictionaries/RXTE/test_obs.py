# This is an example of an RXTE analysis

from Modules import tools as tools

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

# code for the driver, reading in the data files:
# timeRate = np.load(cwd + "/Data/RXTE/obsid_test/test_ch_7_9/7_9_timeRate.npy")  # note that this is flipped from what Jacob will do
# ampCenters = np.load(cwd + "/Data/RXTE/obsid_test/test_ch_7_9/7_9_ampCenters.npy")
# unnattenuated rate

test_obs = {   # BASIC OBSERVATION INFO
            "detector": "RXTE",
            "source_name": "Crab Nebula",
            "source_RA": 83.63426,  # deg
            "source_DEC": 22.010,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.63426), np.deg2rad(22.010)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([100 + 5.55031e8, 385 + 5.55031e8]),  # seconds in MET
            "f107": 75.2,
            "ap": 2,

            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/obsid_test/test_ch_7_9/FPorbit_Day6423",  # NICER orbital solution
}