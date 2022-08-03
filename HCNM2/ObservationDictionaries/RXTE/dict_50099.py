# This is an example of an RXTE analysis

import Modules.tools as tools

import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

# This observation had lock-on troubles
dict_50099 = {   # BASIC OBSERVATION INFO
            "detector": "RXTE",
            "obsID": 50099,
            "epoch": 5,
            "source_name": "Crab Nebula",
            "source_RA": 83.6328,  # deg
            "source_DEC": 22.015,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.6328), np.deg2rad(22.015)),
            "hc_type": "setting",

            # USER-DEFINED DATA
            "crossing_time_range": np.array([227260850, 227261295]),  # seconds in MET
            "t_mid_datetime": '2001-03-15T8:4:32',  # Not used, just for records
            "f107": 134.7,
            "Ap": 2,
            "SN": 110,

            # PATHS TO DATA FILES
            "rossi_path": cwd + "/Data/RXTE/50099/FPorbit_Day2630",  # orbital solution
}
