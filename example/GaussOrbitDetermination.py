import numpy as np

from astro.prelimorbit import Gauss
from astro.time import sidereal_time, julian_date

if __name__ == '__main__':
    mu = np.double(3.986e5)

    sidereal_1 = sidereal_time(20, 8, 2007, 11, 30, 0.0, -110)
    sidereal_2 = sidereal_time(20, 8, 2007, 11, 40, 0.0, -110)
    sidereal_3 = sidereal_time(20, 8, 2007, 11, 50, 0.0, -110)
    julian_1 = julian_date(20, 8, 2007, 11, 30, 0.0)
    julian_2 = julian_date(20, 8, 2007, 11, 40, 0.0)
    julian_3 = julian_date(20, 8, 2007, 11, 50, 0.0)

    print("Sidereal 1:\t\t", sidereal_1)
    print("Sidereal 2:\t\t", sidereal_2)
    print("Sidereal 3:\t\t", sidereal_3)
    print("Julian Date 1:\t", julian_1)
    print("Julian Date 2:\t", julian_2)
    print("Julian Date 3:\t", julian_3)

    gauss = Gauss(mu=mu, use_gibbs=True, improvement=True, improvement_error=10e-15)
    gauss.def_observation(40, -110, 2, degrees=True)
    gauss.add_observation(-33.0588410, -7.2056382, sidereal_1, 0)
    gauss.add_observation(-0.4172870, 17.4626616, sidereal_2, 10 * 60)
    gauss.add_observation(55.0931551, 36.5731946, sidereal_3, 20 * 60)
    r1, r2, r3, v2 = gauss.solve()

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)
    v2_mag = np.linalg.norm(v2)
    print("Radius 1:\t", r1_mag)
    print("Radius 2:\t", r2_mag)
    print("Radius 3:\t", r3_mag)
    print("Velocity 2:\t", v2_mag)

    h_m = np.cross(r2, v2)

    a = mu / ((2 * mu / r2_mag) - np.square(v2_mag))
    h_m_mag = np.linalg.norm(h_m)
    p = np.square(h_m_mag) / mu
    e_mag = np.sqrt(1 - p / a)
    e = np.multiply(np.square(v2_mag) / mu - 1 / r2_mag, r2) - np.multiply(np.dot(r2, v2) / mu, v2)
    n = np.cross([0, 0, 1], h_m)
    n_mag = np.linalg.norm(n)
    i = np.arccos(np.dot([0, 0, 1], h_m) / h_m_mag)
    if i > 180:
        i = 2 * np.pi - i
    big_omega = np.arccos(np.dot([1, 0, 0], n) / n_mag)
    if (n[1] > 0 and big_omega > np.pi) or (n[1] < 0 and big_omega < np.pi):
        big_omega = 2 * np.pi - big_omega
    little_omega = np.arccos(np.dot(n, e) / (n_mag * e_mag))
    if (e[2] > 0 and little_omega > np.pi) or (e[2] < 0 and little_omega < np.pi):
        little_omega = 2 * np.pi - little_omega
    nu0 = np.arccos(np.dot(e, r2) / (e_mag * r2_mag))
    if (np.dot(r2, v2) > 0 and nu0 > 0) or (np.dot(r2, v2) < 0 and nu0 < 0):
        nu0 = np.pi - nu0

    print()
    print("a:\t\t", a)
    print("p:\t\t", p)
    print("e:\t\t", e_mag)
    print("i:\t\t", np.degrees(i))
    print("Omega:\t", np.degrees(big_omega))
    print("omega:\t", np.degrees(little_omega))
    print("nu:\t\t", np.degrees(nu0))
    print()
