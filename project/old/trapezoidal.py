from typing import List

import numpy as np
import matplotlib.pyplot as plt
from algorithms import newton

g = 9.81  # m/s^2
l = 1  # m
m = 1  # kg

def pendulum(_t, y):
    """Compute the two first-order ODEs of a pendulum."""
    return np.array([y[1], -g/l * np.sin(y[0])])

def solve_trapezoidal(func, t0, te, y0, n):
    """General implementation of the trapezoidal rule for solving initial value problems"""
    t = np.linspace(t0, te, num=n)
    y = np.zeros((n, 2))
    h = (te - t0) / (n - 1)  # Step size
    y[0, :] = y0

    for i in range(n - 1):
        # This lambda is the actual trapezoidal rule
        f = lambda yn: y[i] + h/2 * (func(t[i], y[i]) + func(t[i + 1], yn)) - yn
        # Use the Newton's iteration to find a root of F
        # (using a numerical method to compute the Jacobian matrix)
        y[i + 1, :] = newton(f, y[i, :])

    return t, y

def solve_trapezoidal_pendulum(t0, te, y0, n):
    """A specific implementation of the trapezoidal algorithm for a pendulum."""
    t = np.linspace(t0, te, num=n)
    y = np.zeros((n, 2))
    h = (te - t0) / (n - 1)  # Step size
    y[0, :] = y0

    for i in range(n - 1):
        # define the function F to solve F=0
        f = lambda yn: np.array([
            + h / 2 * (y[i, 1] + yn[1]),
            - h * g / (2 * l) * (np.sin(y[i, 0]) + np.sin(yn[0]))
        ]) + y[i, :] - yn
        # define the jacobian matrix function
        j = lambda yn: np.array([[-1, h/2], [-h * g / (2 * l) * np.cos(yn[0]), -1]])
        # Use the Newton's iteration to approximate the
        # root of F (using the exact Jacobian matrix)
        y[i + 1, :] = newton(f, y[i, :], jacobian=j)

    return t, y

def solve_trapezoidal_until(t0, te, y0, h, ao):
    """Implementation of the trapezoidal algorithm for a pendulum with an obstacle at a provided angle a_obst"""
    n = (te - t0) // h + 1
    t = np.zeros(n)
    y = np.zeros((n, 2))
    y[0, :] = y0

    i = 0
    while t[i] < te:
        t[i+1] = t[i] + h
        f = lambda yn: np.array([
            + h / 2 * (y[i, 1] + yn[1]),
            - h * g / (2 * l) * (np.sin(y[i, 0]) + np.sin(yn[0]))
        ]) + y[i, :] - yn
        j = lambda yn: np.array([[-1, h / 2], [-h * g / (2 * l) * np.cos(yn[0]), -1]])
        y[i+1, :] = newton(f, y[i, :], jacobian=j)

        if i == 0:
            i += 1
            continue  # not enough data points to interpolate, so we have to skip

        # create interpolation polynomials of degree 2
        p_a = np.poly1d(np.polyfit(t, y[i-1:i+2, 0], deg=2))
        p_w = np.poly1d(np.polyfit(t, y[i-1:i+2, 1], deg=2))

        # check if any of the roots of p_a-ao are real
        if np.any(np.isreal(roots := (p_a - ao).roots)):
            # check if the roots are in the current time interval [t_i, t_(i+1)]
            in_interval: np.ndarray = (t[i] <= roots) & (roots <= t[i+1])
            if np.any(in_interval):
                t_obst = roots[in_interval][0]
                # update the next values (at i+1) to be at t_obst
                t[i+1] = t_obst
                y[i+1, :] = np.array([ao, p_w(t_obst)])
                # and return all computed steps (discard all other values)
                return t[:i+2], y[:i+2, :]
        i += 1
    return t[:i+1], y[:i+1, :]

def solve_trapezoidal_with_obstacle(t0, te, y0, h, ao):
    pieces: List[np.ndarray] = [np.array([[t0, y0]])]
    while pieces[-1][-1, 0] < te:
        t, y = pieces[-1]
        tn, yn = solve_trapezoidal_until(
            t0=t[-1],
            te=te,
            y0=[y[-1, 0], -y[-1, 1]],
            h=h,
            ao=ao
        )
        pieces.append(np.array([tn, yn]))
    return pieces

def plot_simple(t0, te, y0, n):
    t, y = solve_trapezoidal(pendulum, t0, te, y0, n)
    a = y[:, 0]
    w = y[:, 1]

    plt.plot(t, a, label="$\\alpha$")
    plt.plot(t, w, label="$\\omega$")
    plt.legend()
    plt.grid()
    plt.show()

def plot_energy(t0, te, y0):

    plt.figure(figsize=(10, 6))
    for n in [50, 100, 200, 1000]:
        t, y = solve_trapezoidal(pendulum, t0, te, y0, n)
        a = y[:, 0]
        w = y[:, 1]
        KE = 0.5 * m * (l * w) ** 2  # Kinetic energy
        PE = m * g * l * (1 - np.cos(a))  # Potential energy
        E_total = KE + PE
        plt.plot(t, E_total, label=f"Total Energy (E) N={n}")

    plt.legend()
    plt.title("Energy of Pendulum for different step sizes")
    plt.xlabel('Time t')
    plt.ylabel('Energy E')
    plt.grid()
    plt.show()

def plot_different_n(t0: float | int, te: float | int, y0: List[float] | np.ndarray, n: List[float]):
    for i in range(len(n)):
        t, y = solve_trapezoidal(pendulum, t0, te, y0, n[i])
        plt.plot(t, y[:, 0], label=f"N={n[i]}")
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('angle')
    plt.grid()
    plt.show()

def plot_phase_space(t0, te, y0, n):
    """phase plot of the integration"""
    _, y = solve_trapezoidal(pendulum, t0, te, y0, n)
    a = y[:, 0]
    w = y[:, 1]

    num = 33
    full_a = np.linspace(np.min(a) - 1, np.max(a) + 1, num)
    full_w = np.linspace(np.min(w) - 4, np.max(w) + 4, num)

    A, W = np.meshgrid(full_a, full_w)

    # partial derivatives at every point
    dA = W
    dW = -g / l * np.sin(A)

    plt.figure(figsize=(8, 8))
    plt.plot(a, w, label=f"Phase plot (N={n})")
    plt.streamplot(A, W, dA, dW)
    plt.legend()
    plt.title("Phase space of Pendulum")
    plt.xlabel('angle [rad]')
    plt.ylabel('angular velocity [rad/s]')
    plt.grid()
    plt.show()

def plot_obstacle_simple(t0, te, y0):
    pieces: List[np.ndarray] = solve_trapezoidal_with_obstacle(t0, te, y0, 1/100, -np.pi/6)
    for [t, y] in pieces:
        a = y[:, 1]
        w = y[:, 2]
        plt.plot(t, a, color="r")
        plt.plot(t, w, color="b")

def main() -> None:
    y0 = [np.pi/2, 0]
    t0 = 0
    te = 5
    n = 200

    # without obstacle
    plot_simple(t0, te, y0, n)
    plot_energy(t0, te, y0)
    plot_phase_space(t0, te, y0, n)
    plot_different_n(t0, te, y0, [10, 25, 50, 100])

    # TODO: with obstacle
    plot_obstacle_simple(t0, te, y0)

if __name__ == '__main__':
    main()
