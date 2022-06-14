# This is an example of an RXTE analysis

from Modules import tools as tools

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

test_obs = {   # BASIC OBSERVATION INFO
            "detector": "RXTE",
            "source_name": "Crab Nebula",
            "source_RA": 83.63317,  # deg
            "source_DEC": 22.01453,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.63317), np.deg2rad(22.01453)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([100 + 5.55031e8, 1000 + 5.55031e8]),  # seconds in MET
            "f107": 75.2,
            "ap": 2,

            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/obsid_test/test_ch_7_9/FPorbit_Day6423",  # NICER orbital solution
}