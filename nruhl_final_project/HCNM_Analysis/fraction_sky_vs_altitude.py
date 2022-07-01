# This script makes a plot of F(H) for the HCNM2 paper

import numpy as np
import matplotlib.pyplot as plt

R_earth = 6371  # km
# For the most forgiving model of theta_max, I think that we would want to use the polar radius of the earth: 6356.752 km
def F(H, R_planet=R_earth):
    atan_term = 0.5*np.arctan2(R_planet, np.sqrt((H+R_planet)**2-R_planet**2))
    F = 1 - 2*np.sin((np.pi/4) - atan_term)**2
    return F

def theta_max(H, R_planet=R_earth):
    theta = np.arctan2(R_planet, np.sqrt((R_planet+H)**2-R_planet**2))
    return theta

H = np.arange(300, 3010, 10)

plt.figure(1)

plt.plot(H, F(H))
plt.ylabel(r"Fraction of useful sky, $F$")
plt.xlabel(r"Orbital altitude, $H$ (km)")
plt.grid()

plt.figure(2)
plt.plot(H, np.rad2deg(theta_max(H)))
plt.ylabel(r"Maximum out-of-plane angle, $\theta_{max}$ (degrees)")
plt.xlabel(r"Orbital altitude, $H$ (km)")
plt.grid()

plt.show()


