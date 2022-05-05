import numpy as np
import collections

G = 6.67*10**(-11)
c = 299792458
Msun = 1.988 * 10**30
ly = 60*60*24*365 * c

# Inner most stable circular orbit (ISCO) parameters: 
# https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit
z1 = lambda a: 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
z2 = lambda a: np.sqrt(3 * a**2 + z1(a)**2)
isco_pro = lambda a: (3 + z2(a) - np.sqrt((3 - z1(a)) * (3 + z1(a) + 2*z2(a))))
isco_retro = lambda a: (3 + z2(a) + np.sqrt((3 - z1(a)) * (3 + z1(a) + 2*z2(a))))
keplerian_period = lambda r, a, M: (r**(3/2) + a) *  (2*np.pi * G*M/c**3) / 3600.0  # [hours]

# Angular constants
hour = np.deg2rad(15.0)
radperas = np.deg2rad(1.0/3600.0)
radperuas = radperas*1.e-6
M_to_uas = lambda M, distance: (G*M/c**2 / distance) / radperuas

# SgrA constants / fields
black_hole = collections.namedtuple('black_hole', ['mass', 'distance', 'period'])
sgra = black_hole(
    mass=4.154*10**6 * Msun, 
    distance=26673 * ly, 
    period=lambda r, a: keplerian_period(r, a, sgra.mass),
)