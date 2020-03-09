from astro import prelimorbit


if __name__ == '__main__':
    r0_eci = [1131.340, -2282.343, 6672.423]  # initial radius vector (km)
    v0_eci = [-5.64305, 4.30333, 2.42879]  # initial velocity vector (km/s)
    delta_t = 40 * 60  # time elapsed (s)

    laguerre = prelimorbit.Laguerre(mu=3.986e5)
    r, v = laguerre.solve(r0_eci, v0_eci, delta_t)
