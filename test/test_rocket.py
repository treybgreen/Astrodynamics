import unittest

import numpy as np

import astro


class MyTestCase(unittest.TestCase):
    def test_gravityturn(self):
        rocket = astro.GravityTurn()
        rocket.def_payload(6800)  # kg
        rocket.def_stage(np.double(260.0), np.double(0.10), 0)
        rocket.def_stage(np.double(450.0), np.double(0.12), 1)

        print(rocket.stage[0].epsilon)
        print(rocket.stage[1].epsilon)


if __name__ == '__main__':
    unittest.main()
