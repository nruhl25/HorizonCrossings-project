# This is an example of an RXTE analysis

import Modules.tools as tools

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

dict_40805 = {   # BASIC OBSERVATION INFO
            "detector": "RXTE",
            "obsID": 40805,
            "epoch": 5,
            "source_name": "Crab Nebula",
            "source_RA": 83.63426,  # deg
            "source_DEC": 22.015,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.63426), np.deg2rad(22.015)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([164888000, 164888350]),  # seconds in MET
            "t_mid_datetime": 'NONE',  # Not used, just for records
            "f107": 107.6,
            "Ap": 4,
            "SN": 62,


            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/40805/FPorbit_Day1908",  # orbital solution
}
