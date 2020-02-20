from astro import prelimorbit, util

if __name__ == '__main__':
    r1 = [0, 0, 6378.137]
    r2 = [0, -4464.696, -5102.509]
    r3 = [0, 5740.323, 3189.068]

    gibbs = prelimorbit.Gibbs()
    v2 = gibbs.solve(r1, r2, r3)
    v2_mag = util.magnitude(v2)

    print("v2:", v2)
    print("Magnitude:", v2_mag)
