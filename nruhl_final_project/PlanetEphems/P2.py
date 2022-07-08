# Author: Nathaniel Ruhl
# This is a made-up planet with the size of Earth, but a thinner atmosphere

P2 = {
    # General Info
    "Mass": 5.972 * 10 ** 24,   # kg, mass
    "Radius": 6378.137,  # km, semi-major axis (equatorial radius)
    "surface_gravity": 9.81,  # m/s^2, acceleration due to gravity at sea level

    # Density Profile
    "surface_density": 0.000001225,  # g/cm^3
    "scale_height": 2,  # km, scale height for density exponential

    # Volumetric mix of the atmosphere
    "mix_N": 0.78,
    "mix_O": 0.21,
    "mix_Ar": 0.01,
    "mix_C": 0.0
}

if __name__ == '__main__':
    print(P2)
