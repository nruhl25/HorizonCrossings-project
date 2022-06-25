# This is an example of an RXTE analysis

import Modules.tools as tools

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

dict_50098 = {   # BASIC OBSERVATION INFO
            "detector": "RXTE",
            "obsID": 50098,
            "epoch": 5,
            "source_name": "Crab Nebula",
            "source_RA": 83.6357,  # deg
            "source_DEC": 22.009,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.6357), np.deg2rad(22.009)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([206414281, 206414616]),  # seconds in MET
            "t_mid_datetime": '2000-07-17T1:20:48',  # Not used, just for records
            "f107": 235.8,
            "Ap": 8,
            "SN": 322,

            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/50098/FPorbit_Day2389",  # orbital solution
}
