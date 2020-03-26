import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def three_body(t, x, mu):
    x1, y1, x2, y2 = x

    r1_3 = np.power(np.sqrt(np.square(x1 - mu) + np.square(y1)), 3)
    r2_3 = np.power(np.sqrt(np.square(x1 - 1 + mu) + np.square(y1)), 3)

    x1_dot = x2
    y1_dot = y2
    x2_dot = -(1 - mu) * (x1 - mu) / r1_3 - (mu * (x1 + 1 - mu)) / r2_3 + 2 * y2 + x1
    y2_dot = -((1 - mu) * y1) / r1_3 - mu * y1 / r2_3 - 2 * x2 + y1
    x_dot = [x1_dot, y1_dot, x2_dot, y2_dot]
    return x_dot


if __name__ == '__main__':
    # Constants
    mu = 0.012144  # Earth - Moon Mass/Distance Ratio
    earth_eq_rad = 6378  # Earth Equatorial Radius (km)
    lunar_eq_rad = 1738  # Lunar Equatorial Radius (km)
    r_moon = 384400  # Moon Orbit Radius (km)
    r_earth = r_moon * (mu / (1 - mu))
    dist_earth_moon = r_moon + r_earth
    omega_moon = 2.6491e-6  # Angular Velocity of the Moon (rad/s)
    mu_earth = 3.986e5  # Earth Gravity Constant (km^3/s^2)

    # Targets
    h_leo = 395  # LEO Altitude (km)
    h_periselenium = 395  # Periselenium Altitude (km)
    h_return_perigee = 150  # Earth Return Orbit Altitude (km)

    # Initial Conditions
    departure_speed = np.array([10.75, 10.975])  # Acceptable Departure Speeds (km/s)
    gamma_0 = np.array([75, 150])  # Departure Angle Range (degrees)
    r0 = earth_eq_rad + h_leo

    # Normalized Starting Point
    gamma_init = gamma_0[0]
    velocity_init = departure_speed[0]

    x1_0 = ((-np.cos(np.radians(gamma_init)) * r0) - mu_earth) / dist_earth_moon  # Initial X (km)
    y1_0 = np.sin(np.radians(gamma_init)) * r0 / dist_earth_moon  # Initial Y (km)
    x2_0 = -0.6  # Initial Vx (km/s)
    y2_0 = 0.8  # Initial Vy (km/s)

    # ODE Inputs
    t = (0, 100)  # Time range
    x0 = [x1_0, y1_0, x2_0, y2_0]  # [x, y, x_dot, y_dot] (In the Rotating Frame)

    # Solve ODE
    sol = solve_ivp(three_body, t, x0, 'RK45', args=(mu,))
    print(sol.message)
    print("Number of Iterations:", sol.nfev)
    # print(sol.)
    x = sol.y[0]
    y = sol.y[1]

    # Plot Orbit
    plt.figure(1)
    plt.plot(x, y)
    plt.grid("on")
    plt.title("Orbit")
    earth = plt.Circle((r_earth / dist_earth_moon, 0), earth_eq_rad / dist_earth_moon, color='g')
    plt.gca().add_artist(earth)
    plt.show()

    # Debug
    print(earth_eq_rad / dist_earth_moon)
