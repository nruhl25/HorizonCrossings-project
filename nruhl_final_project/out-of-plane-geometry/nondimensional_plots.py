# This script makes a plot of theta max against the non-dimensional orbital radius
import numpy as np
import matplotlib.pyplot as plt

R_earth = 6378   # km

def theta_max(R_orbit, R_planet):
    return np.arcsin(R_planet/R_orbit)

def F(R_orbit, R_planet):
    F = 1 - 2*np.sin((np.pi/4)-0.5*theta_max(R_orbit, R_planet))**2
    return F

# Uncertainty in the r0 point when
# h_star = h(t-t0)/R_planet
def Delta_r0star(h_star, R_orbit, R_planet):
    return R_orbit*h_star/(R_orbit*np.cos(theta_max(R_orbit, R_planet)))

Rorb_list = np.arange(1,6,0.0001)
hstar_list = np.linspace(0.0001, 0.001, 5000)

plt.figure(1)
plt.plot(Rorb_list, np.rad2deg(theta_max(Rorb_list, 1)))
plt.ylabel(r"Maximum out-of-plane angle, $\theta_{{max}}$ (deg)")
plt.xlabel(r"Non-dimensional orbital radius, $R^* = \frac{R_{{orbit}}}{R_{{planet}}}$")
plt.grid()

plt.figure(2)
plt.plot(Rorb_list, F(Rorb_list, 1))
plt.ylabel(r"Fraction of useful sky $F$")
plt.xlabel(
    r"Non-dimensional orbital radius $R^* = \frac{R_{{orbit}}}{R_{{planet}}}$")
plt.grid()

plt.figure(3)
for R_orb in np.arange(1.05, 1.65, 0.1):
    plt.plot(hstar_list, Delta_r0star(hstar_list, R_orb, 1), label=fr"$R^*$={R_orb:.2f}")

plt.ylabel(r"Non-dimensional in-track error $\Delta r_0^*$")
plt.xlabel(r"Non-dimensional uncertainty in reference altitude $h^*$")
plt.legend()

# Plot specific for and Earth orbit at 420 km
h_list = np.linspace(0, 0.2, 1000)  # km
plt.figure(4)
plt.plot(h_list*1000, 1000*R_earth*Delta_r0star(h_list/R_earth,
                                      R_earth+420, R_earth))  # R_earth*hstar_list
plt.xlabel("Uncertainty in tangent point altitude due to topography (m)")
plt.ylabel("Geometrical in-track error (m)")
plt.grid()


plt.show()
