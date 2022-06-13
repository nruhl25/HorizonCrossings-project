# Author: Nathaniel Ruhl
# The class in this script defines parameters of the central body
# Can also contain information about the atmosphere

G = 6.6743*10**(-11)     # Nm^2/kg^2, Gravitational constant

# For now the default central body "cb" is Earth, and others can be added in the future (read from a ephemeris file, etc...)
class Planet():
    def __init__(self, cb):
        self.cb = cb
        # Import the Planet dictionary into the local namespace, then access with locals() later.
        # There is probably a more concise solution with os.listdir()
        if self.cb == "Earth":
            from PlanetEphems.Earth import Earth
        elif self.cb == "Mars":
            from PlanetEphems.Mars import Mars
        elif self.cb == "Venus":
            from PlanetEphems.Venus import Venus
        elif self.cb == "P1":
            from PlanetEphems.P1 import P1
        elif self.cb == "P2":
            from PlanetEphems.P2 import P2
        else:
            raise RuntimeError("The Planet class is not defined for the user-input")
        # Default properties of the Planet's atmosphere defined below. Atmospheric mix and scalel height can be can be changed internally with property setter methods
        self.planet = locals()[cb]

        self.M = self.planet["Mass"]
        self.R = self.planet["Radius"]
        self.g = self.planet["surface_gravity"]
        self._mix_N = self.planet["mix_N"]
        self._mix_O = self.planet["mix_O"]
        self._mix_Ar = self.planet["mix_Ar"]
        self._mix_C = self.planet["mix_C"]
        self._rho0 = self.planet["surface_density"]
        self._scale_height = self.planet["scale_height"]

    @property
    def mix_N(self):
        return self._mix_N

    @mix_N.setter
    def mix_N(self, mix_N):
        self._mix_N = mix_N

    @property
    def mix_O(self):
        return self._mix_O

    @mix_O.setter
    def mix_O(self, mix_O):
        self._mix_O = mix_O

    @property
    def mix_Ar(self):
        return self._mix_Ar

    @mix_Ar.setter
    def mix_Ar(self, mix_Ar):
        self._mix_Ar = mix_Ar
    
    @property
    def mix_C(self):
        return self._mix_C

    @mix_C.setter
    def mix_C(self, mix_C):
        self._mix_C = mix_C

    @property
    def scale_height(self):
        return self._scale_height

    @scale_height.setter
    def scale_height(self, scale_height):
        self._scale_height = scale_height

    def reset_scale_height(self):
        return self.planet["scale_height"]

    @property
    def rho0(self):
        return self._rho0

    @rho0.setter
    def rho0(self, rho0):
        self._rho0 = rho0
    
    def reset_rho0(self):
        return self.planet["surface_density"]


if __name__ == '__main__':
    Mars = Planet(cb="Mars")


