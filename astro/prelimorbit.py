from matplotlib import pyplot as plt
from scipy import optimize

from astro.util import *


def gauss_q1(x, a, b, c):
    return np.power(x, 8) + a * np.power(x, 6) + b * np.power(x, 3) + c


def gauss_q_improve(x, r2, alpha, rv_r, mu, tau):
    lhs = np.sqrt(mu) * tau
    rhs1 = r2 * rv_r * np.square(x) * stumpff_c(alpha * np.square(x)) / np.sqrt(mu)
    rhs2 = (-alpha * r2 + 1) * np.power(x, 3) * stumpff_s(alpha * np.square(x))
    rhs3 = r2 * x
    return rhs1 + rhs2 + rhs3 - lhs


class Gauss:
    def __init__(self, use_gibbs=True, improvement=True, mu=np.double(3.986e5), flattening=np.double(3.353e-3),
                 radius_earth=np.double(6378.137), improvement_error=np.double(10e-12), max_iter=1000):
        print()
        print("Defining Gaussian Preliminary Orbit Determination\n")
        self.mu = mu
        self.radius_earth = radius_earth  # Radius of Earth (km)
        self.flattening = flattening
        self.observations = 0
        self.latitude = np.double(0.0)
        self.longitude = np.double(0.0)
        self.altitude = np.double(0.0)
        self.ascension = []
        self.declination = []
        self.sidereal = []
        self.time = []
        self.rho = []
        self.radius = []

        self.improvement = improvement
        self.error = improvement_error
        self.max_iteration = max_iter
        self.use_gibbs = use_gibbs
        self.gibbs = Gibbs(mu=self.mu)
        self.rho1_change = []
        self.rho2_change = []
        self.rho3_change = []

    def def_observation(self, latitude, longitude, altitude, degrees=False):
        if degrees:
            self.altitude = altitude
            self.latitude = np.radians(latitude)
            self.longitude = np.radians(longitude)
        else:
            self.altitude = altitude
            self.latitude = latitude
            self.longitude = longitude

        print("\tObservation Site Defined")
        print("\t\tNorthward Latitude:\t", np.degrees(self.latitude))
        print("\t\tWestward Longitude:\t", np.degrees(self.longitude))
        print("\t\tAltitude:          \t", self.altitude)
        print()

    def add_observation(self, ascension, declination, sidereal, seconds_elapsed):
        # time: the amount of time since the first observation in seconds.

        if self.observations >= 3:
            raise Exception("The gauss method only needs 3 observations")

        self.ascension.append(np.radians(ascension))
        self.declination.append(np.radians(declination))
        self.sidereal.append(np.radians(sidereal))
        self.time.append(np.double(seconds_elapsed))
        self.observations += 1
        n = self.observations - 1

        radius_x = (self.radius_earth / np.sqrt(1 - (2 * self.flattening - np.square(self.flattening)) * np.square(
            np.sin(self.latitude))) + self.altitude) * np.cos(self.latitude) * np.cos(self.sidereal[n])
        radius_y = (self.radius_earth / np.sqrt(1 - (2 * self.flattening - np.square(self.flattening)) * np.square(
            np.sin(self.latitude))) + self.altitude) * np.cos(self.latitude) * np.sin(self.sidereal[n])
        radius_z = ((np.square(1 - self.flattening) * self.radius_earth) / np.sqrt(
            1 - (2 * self.flattening - np.square(self.flattening)) * np.square(
                np.sin(self.latitude))) + self.altitude) * np.sin(self.latitude)
        self.radius.append(np.array([radius_x, radius_y, radius_z]))

        l_x = np.cos(self.ascension[n]) * np.cos(self.declination[n])
        l_y = np.sin(self.ascension[n]) * np.cos(self.declination[n])
        l_z = np.sin(self.declination[n])

        self.rho.append(np.array([l_x, l_y, l_z]))

        print("\tGauss Observation", self.observations, "Added")
        print("\t\tAscension:\t\t", np.degrees(self.ascension[n]))
        print("\t\tDeclination:\t", np.degrees(self.declination[n]))
        print("\t\tSidereal Time:\t", np.degrees(self.sidereal[n]))
        print("\t\tTime Elapsed:\t", self.time[n])
        print("\t\tPosition X:\t\t", radius_x)
        print("\t\tPosition Y:\t\t", radius_y)
        print("\t\tPosition Z:\t\t", radius_z)
        print("\t\tRho_hat:\t\t", self.rho[n])
        print()

    def solve(self):
        tau1 = self.time[0] - self.time[1]
        tau3 = self.time[2] - self.time[1]
        tau = self.time[2] - self.time[0]

        p1 = np.cross(self.rho[1], self.rho[2])
        p2 = np.cross(self.rho[0], self.rho[2])
        p3 = np.cross(self.rho[0], self.rho[1])

        d0 = np.dot(self.rho[0], p1)
        d11 = np.dot(self.radius[0], p1)
        d12 = np.dot(self.radius[0], p2)
        d13 = np.dot(self.radius[0], p3)
        d21 = np.dot(self.radius[1], p1)
        d22 = np.dot(self.radius[1], p2)
        d23 = np.dot(self.radius[1], p3)
        d31 = np.dot(self.radius[2], p1)
        d32 = np.dot(self.radius[2], p2)
        d33 = np.dot(self.radius[2], p3)

        aa = (-d12 * tau3 / tau + d22 + d32 * tau1 / tau) / d0
        bb = (d12 * (np.square(tau3) - np.square(tau)) * tau3 / tau + d32 * (
                np.square(tau) - np.square(tau1)) * tau1 / tau) / (6 * d0)
        ee = np.dot(self.radius[1], self.rho[1])

        radius2_square = np.square(np.linalg.norm(self.radius[1]))
        a = -np.square(aa) - 2 * aa * ee - radius2_square
        b = -2 * self.mu * bb * (aa + ee)
        c = -np.square(self.mu) * np.square(bb)

        x_small = 0
        x_big = 15000

        print("\tFinding Root of Q1")
        solution = optimize.root_scalar(gauss_q1, (a, b, c), bracket=[x_small, x_big])
        x = solution.root
        print("\t\tIterations:\t", solution.iterations)
        print("\t\tRoot:\t\t", x)
        print()

        rho1_mag = (((6 * (d31 * tau1 / tau3 + d21 * tau / tau3) * np.power(x, 3) + self.mu * d31 * (
                np.square(tau) - np.square(tau1)) * tau1 / tau3)) / (
                            6 * np.power(x, 3) + self.mu * (np.square(tau) - np.square(tau3))) - d11) / d0
        rho2_mag = aa + self.mu * bb / np.power(x, 3)
        rho3_mag = (((6 * (d13 * tau3 / tau1 - d23 * tau / tau1) * np.power(x, 3) + self.mu * d13 * (
                np.square(tau) - np.square(tau3)) * tau3 / tau1)) / (
                            6 * np.power(x, 3) + self.mu * (np.square(tau) - np.square(tau1))) - d33) / d0

        print("\tFinding Rho Magnitudes")
        print("\t\tRho1 Magnitude:\t", rho1_mag)
        print("\t\tRho2 Magnitude:\t", rho2_mag)
        print("\t\tRho3 Magnitude:\t", rho3_mag)
        print()

        print("\tCalculating Radius Vectors")
        r1 = np.add(self.radius[0], np.dot(rho1_mag, self.rho[0]))
        r2 = np.add(self.radius[1], np.dot(rho2_mag, self.rho[1]))
        r3 = np.add(self.radius[2], np.dot(rho3_mag, self.rho[2]))
        print("\t\tr1:\t", r1)
        print("\t\tr2:\t", r2)
        print("\t\tr3:\t", r3)
        print()

        print("\tFinding f and g Coefficients")
        f1 = 1 - self.mu * np.square(tau1) / (2 * np.power(x, 3))
        f3 = 1 - self.mu * np.square(tau3) / (2 * np.power(x, 3))
        g1 = tau1 - self.mu * np.power(tau1, 3) / (6 * np.power(x, 3))
        g3 = tau3 - self.mu * np.power(tau3, 3) / (6 * np.power(x, 3))
        print("\t\tf11:\t", f1)
        print("\t\tf31:\t", f3)
        print("\t\tg11:\t", g1)
        print("\t\tg31:\t", g3)

        if self.use_gibbs:
            print("\tGibbs Method of Calculating Velocity")
            v2 = self.gibbs.solve(r1, r2, r3, test=False)
            print("\t\tv2:\t", v2)
            print()
        else:
            print("\tGauss Method of Calculating Velocity")
            v2 = np.add(np.multiply(-f3, r1), np.multiply(f1, r3)) / (f1 * g3 - f3 * g1)
            print("\t\tv2:\t", v2)
            print()

        if self.improvement:
            print("\tUsing the Self-Consistent Gauss Method Improvement")
            iteration = 0
            error = []

            self.rho1_change.append(rho1_mag)
            self.rho2_change.append(rho2_mag)
            self.rho3_change.append(rho3_mag)

            f11 = f1
            f31 = f3
            g11 = g1
            g31 = g3

            while True:
                iteration += 1
                rho1_prev = rho1_mag
                rho2_prev = rho2_mag
                rho3_prev = rho3_mag
                f11_prev = f11
                f31_prev = f31
                g11_prev = g11
                g31_prev = g31

                r2_mag = np.linalg.norm(r2)
                v2_mag = np.linalg.norm(v2)
                rv_r = np.dot(r2, v2) / r2_mag
                zeta = 2 / r2_mag - np.square(v2_mag) / self.mu

                sol1 = optimize.root_scalar(gauss_q_improve, (r2_mag, zeta, rv_r, self.mu, tau1),
                                            bracket=[-1000, 1000])
                chi1 = sol1.root

                sol3 = optimize.root_scalar(gauss_q_improve, (r2_mag, zeta, rv_r, self.mu, tau3),
                                            bracket=[-1000, 1000])
                chi3 = sol3.root

                f11 = 1 - np.square(chi1) * stumpff_c(zeta * np.square(chi1)) / r2_mag
                f31 = 1 - np.square(chi3) * stumpff_c(zeta * np.square(chi3)) / r2_mag
                g11 = tau1 - np.power(chi1, 3) * stumpff_s(zeta * np.square(chi1)) / np.sqrt(self.mu)
                g31 = tau3 - np.power(chi3, 3) * stumpff_s(zeta * np.square(chi3)) / np.sqrt(self.mu)
                f11_avg = (f11 + f11_prev) / 2
                f31_avg = (f31 + f31_prev) / 2
                g11_avg = (g11 + g11_prev) / 2
                g31_avg = (g31 + g31_prev) / 2
                c1 = g31_avg / (f11_avg * g31_avg - f31_avg * g11_avg)
                c3 = -g11_avg / (f11_avg * g31_avg - f31_avg * g11_avg)

                rho1_mag = (1 / d0) * (-d11 + d21 / c1 - (c3 / c1) * d31)
                rho2_mag = (-d12 * c1 - d32 * c3 + d22) / d0
                rho3_mag = (-c1 * d13 / c3 + d23 / c3 - d33) / d0
                self.rho1_change.append(rho1_mag)
                self.rho2_change.append(rho2_mag)
                self.rho3_change.append(rho3_mag)

                r1 = np.add(self.radius[0], np.dot(rho1_mag, self.rho[0]))
                r2 = np.add(self.radius[1], np.dot(rho2_mag, self.rho[1]))
                r3 = np.add(self.radius[2], np.dot(rho3_mag, self.rho[2]))

                if self.use_gibbs:
                    v2 = self.gibbs.solve(r1, r2, r3, test=False)
                else:
                    v2 = np.add(np.multiply(-f31, r1), np.multiply(f11, r3)) / (f11 * g31 - f31 * g11)

                error.append(np.sqrt(np.square(rho1_mag - rho1_prev) + np.square(rho2_mag - rho2_prev) + np.square(
                    rho3_mag - rho3_prev)))
                if iteration > self.max_iteration:
                    raise Exception("Max Iteration Limit Reached:", iteration - 1, "Iterations")
                elif error[-1] < self.error:
                    print("\t\tConverged With", iteration, "Iterations")
                    print()
                    break
                else:
                    continue

            plt.figure(1)
            plt.plot(error)
            plt.grid("on")
            plt.title("Error")
            plt.xlabel("Iteration #")
            plt.ylabel("Sum of Delta Rho Squared")
            plt.show()
            # plt.savefig("GaussError.png")

            plt.figure(2)
            plt.plot(self.rho1_change)
            plt.plot(self.rho2_change)
            plt.plot(self.rho3_change)
            plt.grid("on")
            plt.legend(["Rho1", "Rho2", "Rho3"])
            plt.title("Delta Rho")
            plt.xlabel("Iteration #")
            plt.ylabel("Rho Values")
            plt.show()
            # plt.savefig("DeltaRho.png")

            print("\tFinal Radius and Velocity Vectors")
            print("\t\tr1:\t", r1)
            print("\t\tr2:\t", r2)
            print("\t\tr3:\t", r3)
            print("\t\tv2:\t", v2)
            print()

        print("Solution Complete")
        print()
        return r1, r2, r3, v2


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

            print("\tVerify Values are about 0:")
            print("\t\t", d1)
            print("\t\t", d2)
            print("\t\t", d3)
            print("\tEnd Verify")

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
