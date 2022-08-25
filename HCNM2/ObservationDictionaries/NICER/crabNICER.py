import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[2])  # HCNM2/ is cwd

from Modules import tools as tools

crabNICER = {   # BASIC OBSERVATION INFO
            "detector": "NICER",
            "source_name": "Crab Nebula",
            "obsID": 4522010103,
            "source_RA": 83.63317,  # deg
            "source_DEC": 22.01453,  # deg
            "starECI": tools.celestial_to_geocentric(np.deg2rad(83.63317), np.deg2rad(22.01453)),
            "hc_type": "rising",

            # USER-ENTERED INFORMATION (Identify in NICER_Analysis.Step1.py and space weather tables)
            "crossing_time_range": np.array([240165000.0, 240165500.0]),  # seconds in MET
            "spectrum_time_range": np.array([240165300.0, 240165400.0]),
            "t_mid_datetime": '2021-08-11T16:31:19.01',
            "f107": 75.2,
            "Ap": 2,
            "SN": 19,
            "TOD": 0.957,

            # PATHS TO DATA FILES (from cwd, HCNM2/)

            "evt_path": cwd + "/Data/NICER/obs4522010103/ni4522010103_0mpu7_cl.evt",  # NICER events file
            "mkf_path": cwd + "/Data/NICER/obs4522010103/ni4522010103.mkf",  # NICER orbital solution

            "aster_path": None  # ASTER Labs orbital solution
}
