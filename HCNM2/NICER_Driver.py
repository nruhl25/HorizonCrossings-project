# Author: Nathaniel Ruhl

# Driver for HCNM process for a NICER observation

# import local modules for HCNM Analysis
from Modules.OrbitModel import OrbitModel
from Modules.LocateR0hc import LocateR0hc
from Modules.ReadEVT import ReadEVT
from Modules.NormalizeSpectrumNICER import NormalizeSpectrumNICER
from Modules.TransmitModel import TransmitModel
from Modules.CurveComparison import CurveComparison
from Modules.weighted_mean_HC import calc_weighted_mean

# import observation dictionaries
from ObservationDictionaries.NICER.crabNICER import crabNICER
from ObservationDictionaries.NICER.v4641NICER import v4641NICER

# import standard libraries
import numpy as np


def main():
    obs_dict = crabNICER
    # e_band = [1.0, 2.0]  # keV
    bin_size = 1.0   # sec
    e_band_array = np.array([[1.0, 2.0],
                             [2.0, 3.0],
                             [3.0, 4.0]])

    # 1) Define orbit model
    r_array, v_array, t_array = OrbitModel.define_orbit_model(obs_dict, "mkf", time_step=0.01)

    # 2) LocateR0hc (must know hc_type here, R_orbit and h_unit defined within the class)
    r0_obj = LocateR0hc(obs_dict, r_array, v_array, t_array)
    t0_model_index, lat_gp, lon_gp = r0_obj.return_orbit_data()
    del r0_obj

    v0 = v_array[t0_model_index]
    orbit_derived_inputs = (r_array, t_array, t0_model_index, lat_gp, lon_gp)

    # Lists of HCNM measurements for each e_band
    t0_e_list = []
    dt_e_list = []
    for e_band in e_band_array:
        # 3) Bin the data.
        # Can also identify hc_type in this step.
        evt_obj = ReadEVT(obs_dict)
        rate_data, time_data, unattenuated_rate = evt_obj.return_crossing_data(e_band, bin_size)

        spec_obj = NormalizeSpectrumNICER(evt_obj, e_band)
        normalized_amplitudes, bin_centers = spec_obj.return_spectrum_data()
        del evt_obj
        del spec_obj

        # 5) RXTE Analysis starts here.
        eband_derived_inputs = (e_band, bin_size, normalized_amplitudes, bin_centers)

        model_obj = TransmitModel(obs_dict, orbit_derived_inputs, eband_derived_inputs)
        transmit_model, time_crossing_model = model_obj.calculate_transmit_model()

        # GENERATE DATA IF NEEDED
        # time_crossing_model = np.arange(0, 300, bin_size)
        # transmit_model = np.exp(-np.exp(-time_crossing_model+20))

        # 6)
        model_and_data_tuple = (time_crossing_model, transmit_model, time_data, rate_data, unattenuated_rate)

        comp_obj = CurveComparison(obs_dict, model_and_data_tuple)
        t0_e, dt_e = comp_obj.t0_e, comp_obj.dt_e
        del comp_obj

        t0_e_list.append(t0_e)
        dt_e_list.append(dt_e)

        print(f"e_band: {e_band[0]} - {e_band[1]} keV results: ")
        print(f"Time at the position r0_hc = {r_array[t0_model_index]}:")
        print(f"Crossing: t0_e = {t0_e} +/- {dt_e} sec")
        print(f"Input Orbit Model: t0 = {t_array[t0_model_index]} sec")
        print(f"Crossing position uncertainty: +/- {np.linalg.norm(v0)*dt_e:.2f} km")
        print("-----------------------")

    t0, dt = calc_weighted_mean(t0_e_list, dt_e_list)
    dr = np.linalg.norm(v0)*dt
    print("Weighted mean results: ")
    print(f"Crossing: t0_e = {t0:.3f} +/- {dt:.3f} sec")
    print(f"Crossing position uncertainty: +/- {dr:.3f} km")
    print(f"Input Orbit Model: t0 = {t_array[t0_model_index]:.3f} sec")


if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
