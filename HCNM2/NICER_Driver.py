# Author: Nathaniel Ruhl

# Driver for HCNM process for a NICER observation

# import local modules for HCNM Analysis
from Modules.LocateR0hc2 import LocateR0hc2
from Modules.ReadEVT import ReadEVT
from Modules.NormalizeSpectrumNICER import NormalizeSpectrumNICER
from Modules.TransmitModel2 import TransmitModel2
from Modules.CurveComparison import CurveComparison
from Modules.weighted_mean_HC import calc_weighted_mean

# Navigational Driver
from Modules.LocateR0hcNav import LocateR0hcNav
from Modules.FitAtmosphere import FitAtmosphere

# import observation dictionaries
from ObservationDictionaries.NICER.crabNICER import crabNICER
from ObservationDictionaries.NICER.v4641NICER import v4641NICER

# import standard libraries
import numpy as np

def NICER_Driver_MSIS(obs_dict, e_band_array):
    bin_size = 1.0   # sec

    # 1) LocateR0hc2

    r02_obj = LocateR0hc2(obs_dict, "mkf")

    # Lists of HCNM measurements for each e_band
    t0_e_list = []
    dt_e_list = []
    for e_band in e_band_array:
        # 2) Bin the data
        evt_obj = ReadEVT(obs_dict)
        rate_data, time_data, unattenuated_rate = evt_obj.return_crossing_data(
            e_band, bin_size)

        spec_obj = NormalizeSpectrumNICER(evt_obj, e_band)
        normalized_amplitudes, bin_centers = spec_obj.return_spectrum_data()
        del evt_obj
        del spec_obj

        eband_derived_inputs = (
            e_band, bin_size, normalized_amplitudes, bin_centers)

        # 3) Calculate the model transmittance curve
        # Must be 2000 to match with Breck 2020 results
        TransmitModel2.set_pymsis_version(2)
        model2_obj = TransmitModel2(r02_obj, eband_derived_inputs)
        transmit_model = model2_obj.transmit_model
        h_list = model2_obj.h_list    # tangent point altitudes (km)
        # Note that this is [0, time_final], not MET
        time_crossing_model = model2_obj.time_crossing_model

        # 4) Curve comparison: calculate t0_e
        model_and_data_tuple = (
            time_crossing_model, transmit_model, time_data, rate_data, unattenuated_rate)

        CurveComparison.set_comp_range([0.01, 0.99])
        comp_obj = CurveComparison(obs_dict, model_and_data_tuple)
        t0_e, dt_e = comp_obj.t0_e, comp_obj.dt_e
        #  del comp_obj

        np.save("./DensityRetrieval/timeTransmit.npy", np.column_stack((time_data, comp_obj.transmit_data)))
        np.save("./DensityRetrieval/timeAlt.npy", np.column_stack((time_crossing_model+r02_obj.t0_model, h_list)))

        t0_e_list.append(t0_e)
        dt_e_list.append(dt_e)

        print(f"e_band: {e_band[0]} - {e_band[1]} keV results: ")
        print(f"Time at r0_hc = {r02_obj.r0_hc}: ")
        print(f"Crossing: t0_e = {t0_e} +/- {dt_e} sec")
        print(f"Input Orbit Model: t0 = {r02_obj.t0_model} sec")
        print(
            f"Crossing position uncertainty: +/- {np.linalg.norm(r02_obj.v0_model)*dt_e:.2f} km")
        print("-----------------------")

    # 5)
    t0, dt = calc_weighted_mean(t0_e_list, dt_e_list)
    dr = 7.6*dt
    print("Weighted mean results: ")
    print(f"Crossing: t0_e = {t0:.3f} +/- {dt:.3f} sec")
    print(f"Crossing position uncertainty: +/- {dr:.3f} km")
    print(f"Input Orbit Model: t0 = {r02_obj.t0_model} sec")
    return 0


# The problem with NICER data in Nav Mode is that I haven't yet defined the 'TOD' field correctly in the obs_dict
def NICER_Driver_Nav(obs_dict, e_band_kev):
    h0_ref=40
    orbit_model='mkf'
    r0_obj = LocateR0hcNav(obs_dict, orbit_model, h0_ref=40)

    bin_size = 1.0
    #2) Bin the NICER data
    evt_obj = ReadEVT(obs_dict)
    rate_data, time_data, unattenuated_rate = evt_obj.return_crossing_data(
        e_band_kev, bin_size)

    spec_obj = NormalizeSpectrumNICER(evt_obj, e_band_kev)
    normalized_amplitudes, bin_centers = spec_obj.return_spectrum_data()
    del evt_obj
    del spec_obj

    eband_derived_inputs = (
        e_band_kev, bin_size, normalized_amplitudes, bin_centers)

    #3) Fit count rate vs h (geocentric tangent altitudes above y0_ref)
    fit_obj = FitAtmosphere(obs_dict, orbit_model, r0_obj,
                            rate_data, time_data, unattenuated_rate, e_band_kev)

    #4) Determine r50 and t50_model by changing h0_ref in LocateR0hcNav
    r50_obj = LocateR0hcNav(obs_dict, orbit_model,
                            h0_ref=fit_obj.h50_fit+h0_ref)
    r50_model = r50_obj.r0_hc
    t50_model = r50_obj.t0_model

    print(f"h50_adm = {fit_obj.h50_predicted}")
    print(f"h50_measured = {fit_obj.h50_fit} +/- {fit_obj.dy50} km")
    print(f"t50_from_h50={fit_obj.t50_fit} +/- {fit_obj.dt50} sec")
    print(f"t50_newton = {fit_obj.get_t50_newton()} sec")
    print(f"t50_model = {t50_model} sec")
    print(f"Delta t50 = {t50_model - fit_obj.t50_fit} sec")
    dt50_slide = fit_obj.get_dt50_slide()
    print(f"dt50_slide={dt50_slide} sec")
    return 0


def main():
    # remember, with crab, you must alter the curve comparison range below
    obs_dict = v4641NICER
    # NICER_Driver_MSIS(obs_dict,  e_band_array=np.array([[1.0, 2.0]]))
    NICER_Driver_Nav(obs_dict, [4.0, 5.0])
    return 0


if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
