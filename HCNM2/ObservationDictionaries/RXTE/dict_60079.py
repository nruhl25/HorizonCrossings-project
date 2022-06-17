# This is an example of an RXTE analysis

from Modules import tools as tools

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

dict_60079 = {   # BASIC OBSERVATION INFO
            "detector": "RXTE",
            "obsID": 60079,
            "epoch": 5,
            "source_name": "Crab Nebula",
            "source_RA": 83.63426,  # deg
            "source_DEC": 22.010,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.63426), np.deg2rad(22.010)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([8400+2.5992e8, 8800+2.5992e8]),  # seconds in MET
            "f107": 75.2,
            "ap": 2,

            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/60079/FPorbit_Day3008",  # orbital solution
}