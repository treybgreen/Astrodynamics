import unittest

from astro import prelimorbit, util


class PreliminaryOrbitTest(unittest.TestCase):
    def test_laguerre(self):
        r0_eci = [20000, -105000, -19000]  # initial radius vector (km)
        v0_eci = [0.9, -3.4, -1.5]  # initial velocity vector (km/s)
        delta_t = 120 * 60  # time elapsed (s)

        laguerre = prelimorbit.Laguerre()
        r, v = laguerre.solve(r0_eci, v0_eci, delta_t)

        self.assertAlmostEqual(r[0], 26337.76271, places=4)
        self.assertAlmostEqual(r[1], -128751.70147, places=4)
        self.assertAlmostEqual(r[2], -29655.89461, places=4)
        self.assertAlmostEqual(v[0], 0.86280, places=4)
        self.assertAlmostEqual(v[1], -3.21160, places=4)
        self.assertAlmostEqual(v[2], -1.46129, places=4)

    def test_gibbs(self):
        r1 = [0, 0, 6378.137]
        r2 = [0, -4464.696, -5102.509]
        r3 = [0, 5740.323, 3189.068]

        gibbs = prelimorbit.Gibbs()
        v2 = gibbs.solve(r1, r2, r3)
        v2_mag = util.magnitude(v2)

        self.assertAlmostEqual(v2[0], 0.00000, places=4)
        self.assertAlmostEqual(v2[1], 5.5311448, places=4)
        self.assertAlmostEqual(v2[2], -5.1918029, places=4)
        self.assertAlmostEqual(v2_mag, 7.58, places=1)


if __name__ == '__main__':
    unittest.main()
