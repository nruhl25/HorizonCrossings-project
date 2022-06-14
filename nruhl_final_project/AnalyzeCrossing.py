# Author: Nathaniel Ruhl
# This class assembles all the "tools" methods to analyze a horizon crossing

import numpy as np

# import local libraries
from Orbit import Orbit
from xsects import BCM
from gaussxw import gaussxwab


class AnalyzeCrossing(Orbit):

    def __init__(self, cb, H, E_kev=4.0):
        Orbit.__init__(self, cb, H)  # instansiates both Orbit and Planet classes
        self._E_kev = E_kev  # default energy, keV
        self._sigma = BCM.get_total_xsect(
            self.E_kev, self.mix_N, self.mix_O, self.mix_Ar, self.mix_C)  # default sigma

    @property
    def E_kev(self):
        return self._E_kev

    @E_kev.setter
    def E_kev(self, E_kev):
        self._E_kev = E_kev

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def reset_sigma(self):
        return BCM.get_total_xsect(
            self.E_kev, self.mix_N, self.mix_O, self.mix_Ar, self.mix_C)

    ## Define functions relevant to the 2d geometry below:

    # Tangent altitude (km) as a function of elevation angle (rad)
    def tan_alt(self, t):
        h = self.R_orbit*np.sin(self.theta+self.elevation(t))-self.R
        return h

    # Relationship between the total length of the line of sight (km) and elevation angle (rad)


    def d_tot(self, t):
        dtot = 2*np.sqrt(self.R_orbit**2 - (self.R+self.tan_alt(t))**2)
        return dtot

    # Relationship between elevation angle (rad) and angular velocity (rad/sec)


    def elevation(self, t):
        epsilon = self.omega*t
        return epsilon

    # Define functions to convert between a point at distance x on the line of sight and an altitude above Earth, z (km).

    def x_to_z(self, x_km, t):
        z = np.sqrt((self.R+self.tan_alt(t))**2+((self.d_tot(t)/2)-x_km)**2)-self.R
        return z


    def z_to_x(self, z_km, t):
        x = (self.d_tot(t)/2) - np.sqrt((self.R+z_km)**2-(self.R+self.tan_alt(t))**2)
        return x

    # Evaluate the density at a single x distance on the LOS

    def rho_vs_x(self, x_km, t):
        z_km = self.x_to_z(x_km, t)  # radial altitude above Earth
        rho = self.rho_vs_z(z_km)   # g/cm^3, mass density
        return rho

    # Exponential density as a function of altitude (km)
    def rho_vs_z(self, z_km):
        rho = self.rho0*np.exp(-z_km/self.scale_height)
        return rho

    # Define functions for integrating a single line of sight at the time t
    # E is a single value in keV

    # This function returns an array of gamma = optical depth per km along the LOS, corresponding to the input a_array
    def gamma_vs_x(self, x_array_km, t):
        gamma_array = self.sigma*self.rho_vs_x(x_array_km, t)  # cm^-1
        gamma_array = gamma_array*10**5  # km^-1
        return gamma_array

    # Recursive function that calculates the area under gamma from a to b, with midpoint c.
    # gamma curve is defined for E_kev and t
    def qstep(self, a, b, tol, t):
        h1 = b - a
        h2 = h1/2
        c = (a+b)/2
        d = (a+c)/2
        e = (c+b)/2
        # evaluate gamma at the desired points
        ga = self.gamma_vs_x(a, t)
        gb = self.gamma_vs_x(b, t)
        gc = self.gamma_vs_x(c, t)
        gd = self.gamma_vs_x(d, t)
        ge = self.gamma_vs_x(e, t)

        # Evaluate Integrals
        I1 = (h1/6)*(ga + 4*gc + gb)
        I2 = (h2/6)*(ga + 4*gd + 2*gc + 4*ge + gb)

        # Euler-Mclaurin error
        epsilon = (I2-I1)/15

        if abs(epsilon) <= tol:
            I = I2 + epsilon   # better estimate of the integral
            return I, a, c, b
        else:
            # Cut b,the upper bound of the integral, in half
            b = c
            return self.qstep(a, b, tol, t)

    # This is the main function that does the adaptive qudrature and returns the total optical depth
    def tau_adaptive_simpson(self, t, tol):
        # Lists for step size and distance along the LOS
        dx_list = []
        x_midpoints = []
        tau_list = []   # keep track of tau along the LOS, will sum at the end

        # First integral is over the entire domain
        a_moving = 0.0  # will update this in the while loop
        b_fixed = self.d_tot(t)/2

        tau = 0
        # Each iteration of the loop goes through one round of adaptive quadrature
        while a_moving < b_fixed:
            tau_i, x_lower, x_mid, x_upper = self.qstep(
                a_moving, b_fixed, tol, t)
            x_midpoints.append(x_mid)
            dx_list.append(x_upper - x_mid)
            tau_list.append(tau_i)
            a_moving = x_upper   # Upper bound of last integral is lower bound of next integral
        tau = 2*sum(tau_list)
        return tau, dx_list, x_midpoints

    # This function calculates optical depth for a line of sight at the time t with simpson's rule
    def tau_simpson(self, t, N):
        dtot_km = self.d_tot(t)   # km, total length of the los

        # Evaluate the integral over the LOS with Simpson's rule
        # tau = sigma*dx_cm*np.sum(self.rho_vs_x(x_array_km[:-1], t)) is left-hand reimann sum
        a = 0
        b = dtot_km/2
        dx_km = (b-a)/N
        x_array_km = np.arange(a, b+dx_km, dx_km)

        gamma_array = self.gamma_vs_x(x_array_km, t)

        s_odd = 0
        s_even = 0
        for i in range(len(x_array_km)):
            if i % 2 == 0:
                s_even += gamma_array[i]
            else:
                s_odd += gamma_array[i]
        tau = (dx_km/3)*(gamma_array[0] + gamma_array[-1] +
                        4*s_odd + 2*s_even)   # value of integral
        return 2*tau

    # This function calculates optical depth for a line of sight at the time t with gaussian quadrature
    def tau_gauss(self, t, N):
        a = 0.0
        b = self.d_tot(t)/2
        h = (b-a)/N
        xlist, wlist = gaussxwab(N, a, b)
        gamma_array = self.gamma_vs_x(xlist, t)   # Optical depth per km
        # Integrate gamma vs x with gaussian quadrature
        tau_gauss = np.sum(wlist*gamma_array)
        return 2*tau_gauss

    # Method to calculate full transmittance curve
    def calc_transmit_curve(self, time_array):
        transmit_array = np.zeros_like(time_array)
        for i, t in enumerate(time_array):
            transmit_array[i] = np.exp(-self.tau_gauss(t, N=10))
        return transmit_array

    # Methods below are used for the formulation in time
    def beta(self, t):
        numerator = 2*self.R_orbit*self.omega*(self.R+self.tan_alt(t))
        denominator = np.sqrt(self.R_orbit**2 - (self.R+self.tan_alt(t))**2)
        beta = numerator/denominator
        return beta

    def kappa(self, t):
        kappa = self.sigma * self.rho0 * np.exp(-self.tan_alt(t)/self.scale_height) * self.beta(t)
        return kappa

    # This is the exponential integral that appears in Newton's method when solving rho0 or L, uses gaussian quadrature with N = 10 points. User input for scale height can over-ride the instance property
    def exp_integral(self, t, scale_height=None):
        N = 10
        a = 0.0
        b = self.d_tot(t)/2
        h = (b-a)/N
        xlist, wlist = gaussxwab(N, a, b)
        if scale_height is None:
            integrand_array = np.exp(-self.x_to_z(xlist, t)/self.scale_height)
        else:
            integrand_array = np.exp(-self.x_to_z(xlist, t)/scale_height)
        # Integrate with gaussian quadrature
        exp_int = np.sum(wlist*integrand_array)
        exp_int *= 10**5   # convert to cm
        return exp_int

# Code to test the class
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    plt.rc("text", usetex=True)
    sat_alt = 420   # km, satellite altitude
    ES = AnalyzeCrossing(cb="Earth", H=sat_alt)
    MS = AnalyzeCrossing(cb="Mars", H=sat_alt)
    VS = AnalyzeCrossing(cb="Venus", H=sat_alt)
    sat_list = [ES, MS, VS]   # Sattelite list around each planet
    E_kev = 4.0
    hstar50_list = []  # List of hstar at 50% transmission
    rho0_list = []   # List of surface densities
    L_list = []   # list of scale heights

    plt.figure(1)
    plt.title(f"Transmission of {E_kev} keV X-rays")
    plt.figure(2)
    plt.title(f"Transmission of {E_kev} keV X-rays")
    for SAT in sat_list:
        time_array = np.arange(0, SAT.time_final+1, 1)
        transmit_array = np.zeros_like(time_array)
        tan_alt_array = np.zeros_like(time_array)
        for i, t in enumerate(time_array):
            transmit_array[i] = np.exp(-SAT.tau_gauss(t, N=100))
            tan_alt_array[i] = SAT.tan_alt(t)
        hstar_array = tan_alt_array / SAT.scale_height
        print(SAT.cb)
        print(f"Period = {SAT.T} sec")
        print(f"tf = {SAT.time_final} sec")
        print(f"rho0 = {SAT.rho0}")

        plt.figure(1)
        plt.plot(tan_alt_array/SAT.scale_height, transmit_array, label=f"{SAT.cb} satellite at H={SAT.H} km")

        plt.figure(2)
        plt.plot(time_array/SAT.T, transmit_array,
                 label=f"{SAT.cb} satellite at H={SAT.H} km")

    plt.figure(1)
    plt.xlabel("$h^*$")
    plt.ylabel("Transmission")
    plt.legend()

    plt.figure(2)
    plt.xlabel("$t$")
    plt.ylabel("Transmission")
    plt.legend()

    plt.show()
