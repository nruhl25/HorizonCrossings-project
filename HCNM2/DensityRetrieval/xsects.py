# Author: Nathaniel Ruhl
# Cross Section formulas created by Balucinska-Church and McCammon (1992) (xsects in units of cm^2/g)

import numpy as np

# Valid energy range is 0.03 keV to 10 keV
# Cross Sections are for elemental Oxygen, Nitrogen, and Argon
# I put these functions in this class mainly for namespacing purposes
class BCM:
    def __init__(self):
        return

    # This function can either takes the energy as an array of values or just a single value. Each of the other functions only take in single values for energy at a time
    @staticmethod
    def get_total_xsect(mean_energy_kev, mix_N, mix_O, mix_Ar, mix_C):
        energy_ev = mean_energy_kev * 1000
        if isinstance(energy_ev, np.ndarray):
            O_xsect = np.array(list(map(BCM.oxygen_xsect, energy_ev)))
            N_xsect = np.array(list(map(BCM.nitrogen_xsect, energy_ev)))
            Ar_xsect = np.array(list(map(BCM.argon_xsect, energy_ev)))
            C_xsect = np.array(list(map(BCM.carbon_xsect, energy_ev)))
            xsect_total = mix_O * O_xsect + mix_N * N_xsect + mix_Ar * Ar_xsect + mix_C * C_xsect
        else:
            xsect_total = mix_O * BCM.oxygen_xsect(energy_ev) + mix_N * BCM.nitrogen_xsect(energy_ev) \
                + mix_Ar * BCM.argon_xsect(energy_ev) + mix_C * BCM.carbon_xsect(energy_ev)
        return xsect_total

    @staticmethod
    def oxygen_xsect(energy_ev):
        Elog = np.log(energy_ev)
        if energy_ev < 531.7:
            X = 2.57264 + (10.9321 * Elog) + \
                (-1.79383*Elog**2) + (0.102619*Elog**3)
        else:
            X = 16.53869 + (0.6428144 + 3.)*Elog - 0.3177744 * \
                Elog**2 + 7.9471897e-3 * (Elog ** 3)
        oxygen_xsect = np.exp(X)/(energy_ev**3)
        return oxygen_xsect

    @staticmethod
    def nitrogen_xsect(energy_ev):
        Elog = np.log(energy_ev)
        if energy_ev < 401.0:
            X = 9.24058 + (7.02985 * Elog) + (-1.08849 * Elog *
                                              Elog) + (0.0611007 * Elog * Elog * Elog)
        else:
            X = -13.0353 + (15.4851 * Elog) + (-1.89502 *
                                               Elog**2) + (0.0769412*Elog**3)
        nitrogen_xsect = np.exp(X)/(energy_ev**3)
        return nitrogen_xsect

    @staticmethod
    def argon_xsect(energy_ev):
        Elog = np.log(energy_ev)
        if energy_ev < 3202.9:
            X = -330.3509 + (267.7433 + 3.) * Elog - 78.90498 * Elog**2 \
                + 10.35983 * (Elog ** 3) - 0.5140201 * (Elog ** 4)
        elif energy_ev < 3202.9:
            X = -5.71870 + (8.85812 * Elog) + (-0.307357 * Elog * Elog) \
                + (0.00169351 * (Elog ** 3)) + (-0.0138134 * (Elog ** 4)) \
                + (0.00120451 * (Elog ** 5))
        else:
            X = 19.1905 + (2.74276 * Elog) + (-0.164603 *
                                              Elog * Elog) + (0.00165895*Elog**3)
        argon_xsect = np.exp(X)/(energy_ev**3)
        return argon_xsect

    @staticmethod
    def carbon_xsect(energy_ev):
        Elog = np.log(energy_ev)
        if energy_ev < 284.0:
            X = 8.74161 + (7.13348*Elog) + (-1.14604*Elog*Elog) + (0.0677044*Elog*Elog*Elog)
        else:
            X = 3.81334 + (8.93626*Elog) + (-1.06905*Elog*Elog) + (0.0422195*Elog*Elog*Elog)

        carbon_xsect = np.exp(X)/(energy_ev**3)
        return carbon_xsect

def main():
    import matplotlib.pyplot as plt
    E_kev = np.linspace(1, 10, 1000)
    plt.figure(1)
    plt.plot(E_kev, BCM.get_total_xsect(E_kev, 1, 0, 0, 0), label = "N")
    plt.plot(E_kev, BCM.get_total_xsect(E_kev, 0, 1, 0, 0), label="O")
    plt.plot(E_kev, BCM.get_total_xsect(E_kev, 0, 0, 1, 0), label="Ar")
    plt.plot(E_kev, BCM.get_total_xsect(E_kev, 0, 0, 0, 1), label="C")
    plt.legend()

    plt.figure(2)
    E_kev = np.linspace(0.4, 1.0, 1000)
    plt.title("Photoelectric Cross Sections below 1 keV")
    plt.plot(E_kev, BCM.get_total_xsect(E_kev, 0, 1, 0, 0), label="Oxygen")
    plt.plot(E_kev, BCM.get_total_xsect(E_kev, 1, 0, 0, 0), label="Nitrogen")
    plt.plot(E_kev, BCM.get_total_xsect(E_kev, 0, 0, 1, 0), label="Argon")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Cross Section ($cm^2/g$)")
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    main()
