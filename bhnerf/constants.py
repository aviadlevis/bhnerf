import numpy as np
from astropy.constants import *
from astropy import units

# Inner most stable circular orbit (ISCO) parameters: 
# https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit
z1 = lambda a: 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
z2 = lambda a: np.sqrt(3 * a**2 + z1(a)**2)
isco_pro = lambda a: (3 + z2(a) - np.sqrt((3 - z1(a)) * (3 + z1(a) + 2*z2(a))))
isco_retro = lambda a: (3 + z2(a) + np.sqrt((3 - z1(a)) * (3 + z1(a) + 2*z2(a))))

# Black hole quantities
GM_c3 = lambda M: G * M / c**3
GM_c2 = lambda M: G * M / c**2

# SgrA constants / fields
sgra_mass = 4.154*10**6 * M_sun
sgra_distance = 26673 * units.lightyear