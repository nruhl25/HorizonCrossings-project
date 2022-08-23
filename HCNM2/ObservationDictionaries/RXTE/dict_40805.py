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
            "source_RA": 83.6330,  # deg
            "source_DEC": 22.0159,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.6330), np.deg2rad(22.0159)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([164888000, 164888350]),  # seconds in MET
            # Not used, just for records
            "t_mid_datetime": "1999-03-24T10:16:15",
            "f107": 107.6,
            "Ap": 4,
            "SN": 62,
            "TOD": 1.409454,   # calculated inatmospheric_dynamics_model.ipynb


            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/40805/FPorbit_Day1908",  # orbital solution
}
