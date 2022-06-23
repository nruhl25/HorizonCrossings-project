# Author: Nathaniel Ruhl
# This script reads the f107 and ap index for the date of the horizon crossing

from Modules import tools

import pandas as pd
import numpy as np
from pathlib import Path
cwd = str(Path(__file__).parents[1])  # HCNM2/ is cwd


# INPUTS: mid_time_crossing = np.mean() of the identified ~300 sec crossing_time_range
# detector = "NICER" or "RXTE"
# OUTPUTS: f10.7 solar flux,

def get_ap_f107_sn(mid_time_crossing, detector):
    if detector == "RXTE":
        t_cross = tools.convert_time_RXTE(mid_time_crossing)  # datetime object
    elif detector == "NICER":
        t_cross = tools.convert_time_NICER(mid_time_crossing)  # datetime object
    else:
        print("Incorrect user input for 'detector'")

    df = pd.read_table(cwd+"/Data/Weather/Kp_ap_Ap_SN_F107_since_1932.txt", sep="\s+", header=39)
    year_array = np.array(df["#YYY"], dtype=int)
    month_array = np.array(df["MM"], dtype=int)
    day_array = np.array(df["DD"], dtype=int)

    indx = np.where((year_array==t_cross.year) & (month_array==t_cross.month) & (day_array==t_cross.day))[0][0]

    # In the future we can parse the minutes and seconds to get ap
    # example: ap4 is the fourth 1/8 of the 24 hour day
    Ap = df["Ap"][indx]
    f107 = df["F10.7adj"][indx]
    SN = df["SN"][indx]
    return Ap, f107, SN


if __name__ == "__main__":
    # USER-DEFINED FOR THE OBSERVATION
    t_range_crossing = np.array([373742653-150, 373742683+150])  # MET
    mid_time_crossing = np.mean(t_range_crossing)

    Ap, f107, SN = get_ap_f107_sn(mid_time_crossing, "RXTE")
    print(f"f107 = {f107}")
    print(f"Ap = {Ap}")
    print(f"SSN = {SN}")
    print("--------------")



