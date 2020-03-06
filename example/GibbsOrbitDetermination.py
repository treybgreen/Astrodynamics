from astro import prelimorbit, util
import math


if __name__ == '__main__':
    r1 = [0, 0, 6378.137]
    r2 = [0, -4464.696, -5102.509]
    r3 = [0, 5740.323, 3189.068]
    r2_mag = util.magnitude(r2)

    gibbs = prelimorbit.Gibbs()
    v2 = gibbs.solve(r1, r2, r3)
    v2_mag = util.magnitude(v2)

    print("v2:", v2)
    print("v2 Magnitude:", v2_mag)

    h_m = util.cross_product(r2, v2)
    h_m_mag = util.magnitude(h_m)
    dot_rv2 = util.dot_product(r2, v2)
    e_0 = ((v2_mag ** 2 / gibbs.mu) - (1 / r2_mag)) * r2[0] - (dot_rv2 / gibbs.mu) * v2[0]
    e_1 = ((v2_mag ** 2 / gibbs.mu) - (1 / r2_mag)) * r2[1] - (dot_rv2 / gibbs.mu) * v2[1]
    e_2 = ((v2_mag ** 2 / gibbs.mu) - (1 / r2_mag)) * r2[2] - (dot_rv2 / gibbs.mu) * v2[2]
    e = [e_0, e_1, e_2]
    e_mag = util.magnitude(e)
    k_hat = [0, 0, 1]
    i_hat = [1, 0, 0]
    n = util.cross_product(k_hat, h_m)
    n_mag = util.magnitude(n)
    a = gibbs.mu / (((2 * gibbs.mu) / r2_mag) - v2_mag ** 2)
    p = a * (1 - (e_mag ** 2))
    i = math.acos(util.dot_product(k_hat, h_m) / h_m_mag)
    big_omega = math.acos(util.dot_product(i_hat, n) / n_mag)
    little_omega = math.acos(util.dot_product(n, e) / (e_mag * n_mag))
    nu = math.acos(util.dot_product(e, r2) / (e_mag * r2_mag))

    print("Angular Momentum", h_m)
    print("Angular Momentum Magnitude", h_m_mag)
    print("Semi-Major Axis (a):", a)
    print("Semilatus Rectum (p):", p)
    print("Eccentricity:", e)
    print("Eccentricity Magnitude:", e_mag)
    print("Incidence Angle:", i * 180 / math.pi)
    print("Longitude of Ascending Node (big omega):", big_omega * 180 / math.pi)
    print("Argument of Periapsis (little omega):", little_omega * 180 / math.pi)
    print("True Anomaly Angle (nu)", nu * 180 / math.pi)
