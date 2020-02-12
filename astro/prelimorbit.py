import math

from astro.util import stumpff_c, stumpff_s


class Laguerre:
    def __init__(self, error=math.pow(10, -8), mu=398600):
        self.error = error
        self.mu = mu
        return

    def solve(self, r, v, delta_t, e=None):
        r0 = math.sqrt(math.pow(r[0], 2) + math.pow(r[1], 2) + math.pow(r[2], 2))
        v0 = math.sqrt(math.pow(v[0], 2) + math.pow(v[1], 2) + math.pow(v[2], 2))
        r_dot_v = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]

        if e is None:
            e_0 = ((v0 ** 2 / self.mu) - (1 / r0)) * r[0] - (r_dot_v / self.mu) * v[0]
            e_1 = ((v0 ** 2 / self.mu) - (1 / r0)) * r[1] - (r_dot_v / self.mu) * v[1]
            e_2 = ((v0 ** 2 / self.mu) - (1 / r0)) * r[2] - (r_dot_v / self.mu) * v[2]
            e = math.sqrt(math.pow(e_0, 2) + math.pow(e_1, 2) + math.pow(e_2, 2))

        a = self.mu / ((2 * self.mu / r0) - math.pow(v0, 2))  # km
        p = a * (1 - math.pow(e, 2))  # semi-lattice parameter (km)

        x0 = math.sqrt(self.mu) * delta_t / math.fabs(a)
        z0 = math.pow(x0, 2) / a

        n = 0
        error = 1
        while error > self.error:
            f = (1 - r0 / a) * stumpff_s(z0) * math.pow(x0, 3) + (r_dot_v / math.sqrt(self.mu)) * stumpff_c(z0) * \
                math.pow(x0, 2) + r0 * x0 - math.sqrt(self.mu) * delta_t
            fp = stumpff_c(z0) * math.pow(x0, 2) + (r_dot_v / math.sqrt(self.mu)) * (
                    1 - stumpff_s(z0) * z0) * x0 + r0 * (
                         1 - stumpff_c(z0) * z0)
            fpp = (1 - r0 / a) * (1 - stumpff_s(z0) * z0) * x0 + (r_dot_v / math.sqrt(self.mu)) * (
                    1 - stumpff_c(z0) * z0)
            delta = 2 * math.sqrt(4 * math.pow(fp, 2) - 5 * f * fpp)
            delta_x = 5 * f / (fp + math.copysign(delta, fp))
            x0 -= delta_x
            z0 = math.pow(x0, 2) / a
            n += 1
            error = math.fabs(math.pow(delta_x, 2) / a)

        print("Laguerre iterations:", n)

        # Calculate r from the converged X and Z values
        r1_mag = stumpff_c(z0) * math.pow(x0, 2) + (r_dot_v / math.sqrt(self.mu)) * (
                    1 - stumpff_s(z0) * z0) * x0 + r0 * (
                         1 - stumpff_c(z0) * z0)
        nu1 = math.acos(((p / r1_mag) - 1) / e)
        print("nu1: ", nu1 * 180 / math.pi, "(degrees)")
        print("r1:  ", r1_mag, "(km)")

        # Calculate r1 and v1 vectors using f and g coefficients
        l_f = 1 - (x0 ** 2 / r0) * stumpff_c(z0)
        l_fd = (math.sqrt(self.mu) / (r1_mag * r0)) * (z0 * stumpff_s(z0) - 1) * x0
        l_g = delta_t - (math.pow(x0, 3) / math.sqrt(self.mu)) * stumpff_s(z0)
        l_gd = 1 - (stumpff_c(z0) * math.pow(x0, 2)) / r1_mag

        # Calculate r1 vector in ECI coordinates
        r_i = l_f * r[0] + l_g * v[0]
        r_j = l_f * r[1] + l_g * v[1]
        r_k = l_f * r[2] + l_g * v[2]

        # Calculate v1 vector in ECI coordinates
        v_i = l_fd * r[0] + l_gd * v[0]
        v_j = l_fd * r[1] + l_gd * v[1]
        v_k = l_fd * r[2] + l_gd * v[2]

        r1 = [r_i, r_j, r_k]
        v1 = [v_i, v_j, v_k]

        print("r1: <", r_i, "i ,", r_j, "j ,", r_k, "k >")
        print("v1: <", v_i, "i ,", v_j, "j ,", v_k, "k >")
        return r1, v1
