from astro.util import *


class Gibbs:
    def __init__(self, mu=398600):
        self.mu = mu

    def solve(self, r1, r2, r3, test=True):
        r1_mag = magnitude(r1)
        r2_mag = magnitude(r2)
        r3_mag = magnitude(r3)

        c12 = cross_product(r1, r2)
        c23 = cross_product(r2, r3)
        c31 = cross_product(r3, r1)

        if test is True:
            norm1 = normalize(r1, mag=r1_mag)
            norm2 = normalize(r2, mag=r2_mag)
            norm3 = normalize(r3, mag=r3_mag)
            u12 = [x / magnitude(c12) for x in c12]
            u23 = [x / magnitude(c23) for x in c23]
            u13 = [x / magnitude(c31) for x in c31]

            d1 = dot_product(u12, norm3)
            d2 = dot_product(u23, norm1)
            d3 = dot_product(u13, norm2)

            print("Verify Values are about 0:")
            print(d1)
            print(d2)
            print(d3)
            print("End Verify")

        n1 = [r1_mag * x for x in c23]
        n2 = [r2_mag * x for x in c31]
        n3 = [r3_mag * x for x in c12]
        numerator = [n1[0] + n2[0] + n3[0], n1[1] + n2[1] + n3[1], n1[2] + n2[2] + n3[2]]
        numerator_mag = magnitude(numerator)

        denominator = [c12[0] + c23[0] + c31[0], c12[1] + c23[1] + c31[1], c12[2] + c23[2] + c31[2]]
        denominator_mag = magnitude(denominator)

        s1 = r1[0] * (r2_mag - r3_mag) + r2[0] * (r3_mag - r1_mag) + r3[0] * (r1_mag - r2_mag)
        s2 = r1[1] * (r2_mag - r3_mag) + r2[1] * (r3_mag - r1_mag) + r3[1] * (r1_mag - r2_mag)
        s3 = r1[2] * (r2_mag - r3_mag) + r2[2] * (r3_mag - r1_mag) + r3[2] * (r1_mag - r2_mag)
        s = [s1, s2, s3]
        s_mag = magnitude(s)

        multiple = math.sqrt(self.mu / (numerator_mag * denominator_mag))

        cdr2 = cross_product(denominator, r2)
        cdr2_r2 = [x / r2_mag for x in cdr2]
        v_over_hm = [cdr2_r2[0] + s[0], cdr2_r2[1] + s[1], cdr2_r2[2] + s[2]]

        v2 = [multiple * x for x in v_over_hm]
        return v2


class Laguerre:
    def __init__(self, error=math.pow(10, -8), mu=398600):
        self.error = error
        self.mu = mu
        return

    def solve(self, r, v, delta_t, e=None):
        r0 = magnitude(r)
        v0 = magnitude(v)
        r_dot_v = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]

        if e is None:
            e_0 = ((v0 ** 2 / self.mu) - (1 / r0)) * r[0] - (r_dot_v / self.mu) * v[0]
            e_1 = ((v0 ** 2 / self.mu) - (1 / r0)) * r[1] - (r_dot_v / self.mu) * v[1]
            e_2 = ((v0 ** 2 / self.mu) - (1 / r0)) * r[2] - (r_dot_v / self.mu) * v[2]
            e = magnitude([e_0, e_1, e_2])

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
                1 - stumpff_s(z0) * z0) * x0 + r0 * (1 - stumpff_c(z0) * z0)
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
