# This is an example of an RXTE analysis

import Modules.tools as tools

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

dict_91802 = {   # BASIC OBSERVATION INFO
            "detector": "RXTE",
            "obsID": 91802,
            "epoch": 5,
            "source_name": "Crab Nebula",
            "source_RA": 83.632,  # deg
            "source_DEC": 22.018,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.632), np.deg2rad(22.018)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([373742653-150, 373742683+150]),  # seconds in MET
            "t_mid_datetime": 'NONE',  # Not used, just for records
            "f107": 76.1,
            "Ap": 20,
            "SN": 18,

            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/91802/FPorbit_Day4325",  # orbital solution
}