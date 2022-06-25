# Author: Nathaniel Ruhl
# This script a new algorith to l"locate r0_hc" that uses non-linear root solving. It can also be used to verify Seamus's maximum out-of-plane angle formula

import numpy as np
import matplotlib.pyplot as plt

# Define constants
G = 6.6743*10**(-11)    # Nm^2/kg^2, Gravitational constant
R_earth = 6371  # km
M_earth = 5.972 * 10 ** 24  # kg, mass of Earth

################ DEFINE DESIRED ORBIT ################
H = 420
R_planet = R_earth
M_planet = M_earth
hc_type = "rising"
h_unit = np.array([0, 0, 1])  # (aka we're alread in the perifocal frame)

# Derived values
R_orbit = R_planet + H
T = np.sqrt(4*np.pi**2/(G*M_planet)) * R_orbit**3
omega = 2*np.pi/T
t_orbit = np.arange(0, T, 1)
#####################################################

# Function to project the source onto the plane of the orbit
def proj_on_orbit(r_source, h_unit):
    r_prime_source = r_source - h_unit * \
        np.dot(r_source, h_unit)   # project on orbit plane
    r_prime_source = r_prime_source / \
        np.linalg.norm(r_prime_source)  # normalize the vector
    return r_prime_source

# position of satelite in orbital model
def r(t):
    if isinstance(t, np.ndarray):
        x = R_orbit*np.cos(omega*t)
        y = R_orbit*np.sin(omega*t)
        z = np.zeros_like(x)
        return np.column_stack((x, y, z))
    else:
        # t is a scalar
        x = R_orbit*np.cos(omega*t)
        y = R_orbit*np.sin(omega*t)
        z = 0.0
        return np.array([x, y, z])

# Function to minimize when identifying r0
# Below, some global variables relevant to function
n_max = 3000  # km, max distance along LOS to look for grazing (should be fully general)
n_list = np.arange(0, n_max, 0.1)   # 0.1 km steps
n_column_vec = n_list.reshape((len(n_list), 1))
def f(t, s_unit):
    # define the los
    starArray = np.ones((len(n_list), 3)) * s_unit
    los = r(t) + n_column_vec * starArray
    p_mag = np.linalg.norm(los, axis=1)
    alt_tp = np.min(p_mag)  # Altitude of tangent point, km (this eqn is slightly different for allipsoid planet in which case R_planet is an array that depends on los polar angles)
    return alt_tp - R_planet

# This function returns r0_hc for an arbitrary out-of-plane angle, psi (deg).
def find_r0(psi_deg):

    psi = np.deg2rad(psi_deg)   # radians
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(psi), -np.sin(psi)],
                    [0, np.sin(psi), np.cos(psi)]])   # Principle rotation about x axis
    s_proj = np.array([0, 1, 0])   # in-plane source on y-axis (of perifocal frame)

    s_unit = np.dot(R_x, s_proj)

    # Define r0_2d
    s_proj = proj_on_orbit(s_unit, h_unit)
    # Use the 2d formulas to guess where r0 may be
    if hc_type == "rising":
        g_unit_proj = np.cross(s_proj, h_unit)
    elif hc_type == "setting":
        g_unit_proj = np.cross(h_unit, s_proj)

    A_2d = np.sqrt(R_orbit ** 2 - R_planet ** 2)
    r0_2d = R_planet * g_unit_proj - A_2d * s_proj

    dr = np.linalg.norm(r(t_orbit) - r0_2d, axis=1)   # list of errors from r0_2d
    t1_index = np.argmin(dr)
    t1 = t_orbit[t1_index]

    # Newton's method to minimize f(t)
    t = t1  # initial guess
    t_last = t1 - 1  # initial guess for secant method

    b = 10  # initialization to enter for loop
    graze_tolerance = 1e-3 # 1 m, initialize graze_tolerance for identifying r0
    num_iter = 1
    while(abs(b)>graze_tolerance):
        b = f(t, s_unit)
        m = (f(t, s_unit) - f(t_last, s_unit))/(t-t_last)
        delta = b/m
        t_last = t
        t -= delta
        num_iter += 1
    
    r0_hc = r(t)
    print(f"psi = {psi_deg} deg")
    print(f"{num_iter} iterations")
    print(f"r0_2d = {r0_2d}")
    print(f"r0_model1 = {r(t1)}")
    print(f"r0_hc = {r0_hc}")
    print("-------------------")
    return r0_hc, s_unit

def main():
    orbit_vec = r(t_orbit)
    psi_list = np.arange(0, 70, 10) # max seems to be 69 for the ISS orbit

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(orbit_vec[:, 0], orbit_vec[:, 1], orbit_vec[:, 2], s=1)
    for psi in psi_list:
        r0_hc, s_unit = find_r0(psi)
        ax.scatter3D(*r0_hc, label=fr"$\psi = ${psi}$^\circ$")
        # ax.quiver(*r0_hc, *s_unit, length = R_orbit) These are not plotting well

    ax.view_init(90,-90)
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    main()
