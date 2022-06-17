# import local modules
from Modules import tools as tools

# import standard libraries
import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

v4641NICER = {   # BASIC OBSERVATION INFO
            "detector": "NICER",
            "source_name": "V4641 Sgr.",
            "obsID": None,
            "source_RA": 274.839,  # deg
            "source_DEC": -25.407,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(274.839), np.deg2rad(-25.407)),
            "hc_type": "rising",

            # USER-ENTERED INFORMATION (Define in NICER_Analysis_Step1.py and space weather tables)
            "crossing_time_range": np.array([300 + 1.92224e8, 760 + 1.92224e8]),  # seconds in MET
            "spectrum_time_range": np.array([550 + 1.92224e8, 690 + 1.92224e8]),
            "f107": 69.7,
            "ap": 12.0,

            # PATHS TO DATA FILES (from cwd, HCNM2/)

            "evt_path": cwd + "/Data/NICER/2-3-20-v4641/NICER_events.evt",  # NICER events file
            "mkf_path": cwd + "/Data/NICER/2-3-20-v4641/ISS_orbit.mkf",  # NICER orbital solution

            "aster_path": None  # ASTER Labs orbital solution
}
