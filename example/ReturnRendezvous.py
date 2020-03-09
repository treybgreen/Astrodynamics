import numpy as np

from astro import golden_search, rotate_2D


def transfer_time_a(a, r1, r2, c, time_offset=np.double(0.0), mu=np.double(3.986e5)):
    alpha = np.arccos(1 - (r1 + r2 + c) / (2 * a))
    beta = np.arccos(1 - (r1 + r2 - c) / (2 * a))
    delta_t = np.float_power(a, 1.5) * (alpha - beta - (np.sin(alpha) - np.sin(beta))) / np.sqrt(mu)
    return np.fabs(delta_t - time_offset)


def transfer_time_p(p, a, r1, r2, k, l, m, delta_nu, time_offset=np.double(0.0), mu=np.double(3.986e5)):
    f = 1 - (r2 / p) * (1 - np.cos(delta_nu))
    f_dot = np.sqrt(mu / p) * np.tan(delta_nu / 2) * (((1 - np.cos(delta_nu)) / p) - (1 / r1) - (1 / r2))
    g = r1 * r2 * np.sin(delta_nu) / np.sqrt(mu * p)
    # g_dot = 1 - (r1 / p) * (1 - np.cos(delta_nu))

    cos_e = 1 - (r1 / a) * (1 - f)
    sin_e = -r1 * r2 * f_dot / np.sqrt(mu * a)

    if sin_e >= 0:
        delta_e = np.arccos(cos_e)
    else:
        delta_e = 2 * np.pi - np.arccos(cos_e)

    delta_t = g + np.sqrt(np.power(a, 3) / mu) * (delta_e - sin_e)
    return np.fabs(delta_t - time_offset)


def p_iteration(a, r1, r2, delta_nu, delta_t, mu=np.double(3.986e5), error=np.double(10e-8)):
    k = r1 * r2 * (1 - np.cos(delta_nu))
    l = r1 + r2
    m = r1 * r2 * (1 + np.cos(delta_nu))

    p_i = k / (l + np.sqrt(2 * m))
    p_ii = k / (l - np.sqrt(2 * m))

    return golden_search(p_i + 10, p_ii - 10, error, transfer_time_p, a, r1, r2, k, l, m, delta_nu, delta_t, mu)


if __name__ == '__main__':
    # Constants
    mu = np.double(3.986e5)

    # Hyperbolic Orbit
    a_h = np.double(-3.0e4)
    e_h = np.sqrt(2)
    p_h = a_h * (1 - np.square(e_h))
    r1_h = p_h

    # Elliptic Orbit
    a_e = np.double(15000)
    e_e = np.double(0.5)
    p_e = a_e * (1 - np.square(e_e))
    r1_e = a_e * (1 - e_e)
    r2 = a_e * (1 + e_e)
    eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - e_e) / (1 + e_e)) * np.tan(np.pi / 2))
    delta_t = np.sqrt(np.power(a_e, 3) / mu) * (eccentric_anomaly - e_e * np.sin(eccentric_anomaly))

    # Transfer Orbit
    delta_nu = np.pi / 2
    c = np.sqrt(np.square(r1_h) + np.square(r2) - 2 * r1_h * r2 * np.cos(delta_nu))

    a = golden_search(25000, 100000, 10e-15, transfer_time_a, r1_h, r2, c, delta_t)
    delta = transfer_time_a(a, r1_h, r2, c, mu=mu)
    v1_t = np.sqrt((2 * mu / r1_h) - (mu / a))
    v2_t = np.sqrt((2 * mu / r2) - (mu / a))

    p = p_iteration(a, r1_h, r2, delta_nu, delta_t, mu=np.double(3.986e5))

    e_magnitude = np.sqrt(1 - p / a)

    # Spacecraft is approaching perigee so it is in the 3rd or 4th quadrant
    nu1_hyperbolic = -np.pi / 2
    nu1_transfer = 2 * np.pi - np.arccos(((p / r1_h) - 1) / e_magnitude)
    nu2_transfer = 2 * np.pi - np.arccos(((p / r2) - 1) / e_magnitude)
    nu2_final = np.pi

    theta = (np.pi - nu2_transfer) % (2 * np.pi)

    e = rotate_2D(theta, np.array([e_magnitude, 0]))

    v1_initial = np.sqrt(mu / p_h) * np.array([-np.sin(nu1_hyperbolic), e_h + np.cos(nu1_hyperbolic)])
    v1_initial_rotated = rotate_2D(np.pi, v1_initial)
    v1_transfer = np.sqrt(mu / p) * np.array([-np.sin(nu1_transfer), e_magnitude + np.cos(nu1_transfer)])
    v1_transfer_rotated = rotate_2D(theta, v1_transfer)
    v2_transfer = np.sqrt(mu / p) * np.array([-np.sin(nu2_transfer), e_magnitude + np.cos(nu2_transfer)])
    v2_transfer_rotated = rotate_2D(theta, v2_transfer)
    v2_final = np.sqrt(mu / p_e) * np.array([-np.sin(nu2_final), e_e + np.cos(nu2_final)])

    delta_v1 = v1_transfer_rotated - v1_initial_rotated
    delta_v1_mag = np.linalg.norm(delta_v1)
    delta_v2 = v2_final - v2_transfer_rotated
    delta_v2_mag = np.linalg.norm(delta_v2)

    print("a:", a)
    print("p:", p)
    print("e:", e_magnitude)
    print("Time:", delta_t)
    print("Difference:", np.fabs(delta_t - delta))
    print("Rotation of Transfer Orbit:", (np.pi - (np.pi * 2 - nu2_transfer)) * 180 / np.pi)
    print("V1:", v1_t)
    print("Nu1:", nu1_transfer * 180 / np.pi)
    print("V2:", v2_t)
    print("Nu2:", nu2_transfer * 180 / np.pi)
    print("Theta:", np.degrees(theta))
    print("Eccentricity:", e)
    print("V1_initial:", v1_initial_rotated)
    print("V1_transfer:", v1_transfer, "Magnitude:", np.linalg.norm(v1_transfer))
    print("V1_transfer Rotated:", v1_transfer_rotated, "Magnitude:", np.linalg.norm(v1_transfer_rotated))
    print("V1_transfer:", v2_transfer, "Magnitude:", np.linalg.norm(v2_transfer))
    print("V2_transfer Rotated:", v2_transfer_rotated, "Magnitude:", np.linalg.norm(v2_transfer_rotated))
    print("V2_final:", v2_final)
    print("Delta V1:", delta_v1, "Magnitude:", delta_v1_mag)
    print("Delta V2:", delta_v2, "Magnitude:", delta_v2_mag)
    print("Total Delta V:", delta_v1_mag + delta_v2_mag)
