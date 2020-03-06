import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

R_earth = 6378 * 1000  # m
mu = np.double(3.986e5) * np.power(1000, 3)  # m^3/s^2
rho0 = np.double(1.225)  # kg/m^3
scale_height = np.double(8000)  # m
cd_inf = np.double(0.155)
gravity0 = np.double(9.80665)  # m/s^2

delta = np.double(0.99999999995)  # Initial Angle of Rocket
beta1 = np.double(-500)  # Fuel Burn Rate of Stage 1
beta2 = np.double(-400)  # Fuel Burn Rate of Stage 2
m1 = np.double(175000)  # kg (mass of Stage 1)
m2 = np.double(50000)  # kg (mass of Stage 2)

m_star = np.double(6800.0)  # kg
epsilon1 = np.double(0.10)
epsilon2 = np.double(0.12)
isp1 = np.double(260.0)  # s
isp2 = np.double(450.0)  # s

rocketArea = np.double(20)  # m^2
x0 = np.double(0.0)  # m
gamma0 = np.double((np.pi / 2) * delta)  # radians
velocity0 = np.double(0.001)  # m/s
height0 = np.double(0.0)  # m

m1s = m1 * epsilon1
m1p = (m1s / epsilon1) - m1s
m2s = m2 * epsilon2
m2p = (m2s / epsilon2) - m2s
m01 = m1 + m2 + m_star
m02 = m2 + m_star

burn1 = -m1p / beta1  # s (Burn time of stage 1)
burn2 = -m2p / beta2  # s (Burn time of stage 2)

beta = beta1
isp = isp1
v_ex = isp * gravity0
m0 = m01


# def cd(mach):
#     if mach > 2:
#         return cd_inf
#     else:
#         return 10 * np.sqrt(cd_inf) * ((mach ** 9) - (1 / 12) * np.sqrt(cd_inf)) * np.exp(-4.2 * (mach ** 3)) + cd_inf


def rho(height):
    return rho0 * np.exp(-height / scale_height)


def gravity(height):
    return mu / ((R_earth + height) ** 2)


def thrust():
    return -beta * v_ex


def drag(mass, velocity, height):
    return np.double(0.5) * rho(height) * (velocity ** 2) * (cd_inf * rocketArea / mass)


def x_dot(velocity, gamma):
    return velocity * np.cos(gamma)


def h_dot(velocity, gamma):
    return velocity * np.sin(gamma)


def gamma_dot(mass, velocity, gamma, height):
    return -(gravity(height) * mass - (mass * (x_dot(velocity, gamma) ** 2)) / (R_earth + height) * np.cos(gamma)) / (
            mass * velocity)


def acceleration(mass, velocity, gamma, height):
    return (thrust() - drag(mass, velocity, height) - (
            gravity(height) * mass - (mass * (x_dot(velocity, gamma) ** 2)) / (R_earth + height)) * np.sin(
        gamma)) / mass


def gravity_turn(t, initial):
    downrange, height, velocity, gamma = initial

    mass = m0 + (beta * t)

    xd = x_dot(velocity, gamma)
    hd = h_dot(velocity, gamma)
    vd = acceleration(mass, velocity, gamma, height)
    gd = gamma_dot(mass, velocity, gamma, height)
    y1 = [xd, hd, vd, gd]
    return y1


if __name__ == '__main__':
    print("\t\t\t\t\tStage 1\t\t\tStage 2")
    print("Structural Mass\t\t", m1s, "\t\t", m2s)
    print("Propellant Mass\t\t", m1p, "\t\t", m2p)
    print("Burn Time\t\t\t", burn1, "\t\t\t", burn2)

    # Mach_Array = np.linspace(0, 10, 200000)
    # CD_Array = cd(Mach_Array)
    # plt.figure(1)
    # plt.plot(Mach_Array, CD_Array)
    # plt.grid("on")
    # plt.title("C_d vs Mach")
    # plt.xlabel("Mach")
    # plt.ylabel("C_d")
    # plt.savefig("Cd_Mach.png")
    #
    # H_Array = np.linspace(0, 100000, 200000)
    # Rho_Array = rho(H_Array)
    # plt.figure(2)
    # plt.plot(H_Array, Rho_Array)
    # plt.grid("on")
    # plt.title("Density vs Height")
    # plt.xlabel("Height")
    # plt.ylabel("Density")
    # plt.savefig("Density_Height.png")

    y0 = [x0, height0, velocity0, gamma0]

    sol = solve_ivp(gravity_turn, (0.0, burn1), y0, 'RK45')
    print(sol.message)
    print(sol.t)
    print(sol.y)

    t = sol.t
    x = sol.y[0]
    h = sol.y[1]
    v = sol.y[2]
    g = sol.y[3]

    plt.figure(3)
    plt.plot(t, x)
    plt.grid("on")
    plt.title("Distance Downrange")
    plt.show()

    plt.figure(4)
    plt.plot(t, h)
    plt.grid("on")
    plt.title("Height")
    plt.show()

    plt.figure(5)
    plt.plot(t, v)
    plt.grid("on")
    plt.title("Velocity")
    plt.show()

    plt.figure(6)
    plt.plot(t, g)
    plt.grid("on")
    plt.title("Flight Path Angle")
    plt.show()
