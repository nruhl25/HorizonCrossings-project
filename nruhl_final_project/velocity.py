# Author: Nathaniel Ruhl
# This script is used to plot the formulas for tangent altitude and velocity

from AnalyzeCrossing import AnalyzeCrossing

import numpy as np
import matplotlib.pyplot as plt

ISS = AnalyzeCrossing("Earth", 420)

print(f"{4.5/(ISS.R_orbit*ISS.omega*np.sin(ISS.theta+40*ISS.omega))}")
t = np.arange(0, 300, 1.0)
h = ISS.R_orbit*np.sin(ISS.theta+ISS.omega*t)-ISS.R
h2 = ISS.R_orbit*np.sin(ISS.theta+(ISS.omega+(0.003*ISS.omega))*t)-ISS.R

# error in measurement due to uncertainty in angular velocity (assume t50 is identified)
# Really only linear if we define h_50 in a library ahead of time.
def Delta_t_linear(sat, time_array, Delta_omega):
    h50, t50 = sat.h50_t50(time_array)
    Delta_t = -t50*Delta_omega*np.cos(sat.theta+sat.omega*t50)/(sat.omega*np.cos(sat.theta))
    return Delta_t

plt.figure(1)
plt.plot(t, h)
plt.plot(t, h2, label="Sat. moving 0.003% faster")
plt.hlines(110, t[0], t[-1], label='110 km', color='r')
plt.ylabel("Tangent Altitude, h (km)")
plt.xlabel("Time (sec)")
plt.legend()

# factor_list = np.linspace(0, 0.1, 100)
# Delta_omega_list = factor_list*ISS.omega
# Delta_t_linear_list = []
# Delta_t_list = []
# for Delta_omega in Delta_omega_list:
#     Delta_t_linear_list.append(Delta_t_linear(ISS, t, Delta_omega))
#     Delta_t_list.append(ISS.h50_t50(t, Delta_omega)[
#                         1]-ISS.h50_t50(t)[1])
# plt.figure(2)
# plt.plot(100*factor_list, Delta_t_list, label=r'$\Delta t_0$')
# plt.plot(100*factor_list, Delta_t_linear_list, label=r'Linearized $\Delta t_0$')
# plt.xlabel("Percent error in orbital velocity (H=420km)")
# plt.ylabel("Error in horizon crossing measurement (sec)")
# plt.legend()


plt.figure(3)
dh_dw = ISS.R_orbit*t*np.cos(ISS.theta+ISS.omega*t)
for N in np.arange(7, 12, 1):
    dw = ISS.omega/N
    h2 = ISS.R_orbit*np.sin(ISS.theta+(ISS.omega+dw)*t)-ISS.R
    plt.plot(t, dh_dw*dw, label=f'{(dw/ISS.omega):.2f}% error, linearized')
    plt.plot(t, h2-h, label=f'{(dw/ISS.omega):.2f}% error, non-linearized')
plt.legend()
plt.show()



