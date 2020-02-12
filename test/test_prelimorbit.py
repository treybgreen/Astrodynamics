import unittest

from astro import prelimorbit


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


if __name__ == '__main__':
    unittest.main()
