from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt


g = 9.81  # m/s^2
l = 1  # m
m = 1  # kg


def newton(
    function: Callable,
    initial_guess: List | np.ndarray,
    jacobian: Callable,
    tolerance: float | int = 1e-5,
    max_iter: int = 1000,
):
    # if jacobian is None:
    #     jacobian = lambda guess: numerical_jacobian(function, guess)
    current_guess = np.array(initial_guess, dtype=float)
    for _ in range(int(max_iter)):
        j = jacobian(current_guess)
        f = function(current_guess)
        update = np.linalg.solve(j, f)
        current_guess -= update
        if np.linalg.norm(update) < tolerance:
            break
    return current_guess


# def numerical_jacobian(func, x, epsilon=1e-6):
#     n = len(x)
#     m = len(func(x))
#     j = np.zeros((m, n))
#     for i in range(n):
#         x_shifted = x.copy()
#         x_shifted[i] += epsilon
#         j[:, i] = (func(x_shifted) - func(x)) / epsilon
#     return j


# ------------------------------------- without obstacle -------------------------------------


# def pendulum(_, y):
#     """Compute the two first-order ODEs of a pendulum."""
#     return np.array([y[1], -g / l * np.sin(y[0])])
#
#
# def solve_trapezoidal(func, t0, te, y0, n):
#     """General implementation of the trapezoidal rule for solving initial value problems"""
#     t = np.linspace(t0, te, num=n)
#     y = np.zeros((n, 2))
#     h = (te - t0) / (n - 1)  # Step size
#     y[0, :] = y0
#
#     for i in range(n - 1):
#         # This function is the actual trapezoidal rule
#         def f(yn):
#             return y[i] + h / 2 * (func(t[i], y[i]) + func(t[i + 1], yn)) - yn
#
#         # Use the Newton's iteration to find a root of F
#         # (using a numerical method to compute the Jacobian matrix)
#         y[i + 1, :] = newton(f, y[i, :])
#
#     return t, y
#
#
# def solve_trapezoidal_pendulum(
#     t0: int | float, te: int | float, y0: int | float | List | np.ndarray, n: int
# ):
#     """A specific implementation of the trapezoidal algorithm for a pendulum."""
#     t = np.linspace(t0, te, num=n)
#     y = np.zeros((n, 2))
#     h = (te - t0) / (n - 1)  # Step size
#     y[0, :] = y0
#
#     # define the jacobian matrix function
#     def j(yn):
#         return np.array([[-1, h / 2], [-h * g / (2 * l) * np.cos(yn[0]), -1]])
#
#     for i in range(n - 1):
#         # define the function F to solve F=0
#         def f(yn):
#             return (
#                 np.array(
#                     [
#                         +h / 2 * (y[i, 1] + yn[1]),  # F_1
#                         -h * g / (2 * l) * (np.sin(y[i, 0]) + np.sin(yn[0])),  # F_2
#                     ]
#                 )
#                 + y[i, :]  # y_n
#                 - yn  # y_(n+1)
#             )
#
#         # Use the Newton's iteration to approximate the
#         # root of F (using the exact Jacobian matrix)
#         y[i + 1, :] = newton(f, y[i, :], jacobian=j)
#
#     return t, y


# ------------------------------------- with obstacle -------------------------------------


def solve_trapezoidal_until(
        t0: float | int,
        te: float | int,
        y0: int | float | List | np.ndarray,
        h: float | int,
        ao: float | int = None
):
    t = np.arange(t0, te + h, step=h)
    # t[-1] = te  # important if te - t0 is not a multiple of h
    y = np.zeros((len(t), 2))
    y[0, :] = y0  # set the initial conditions

    def j(yn):
        return np.array([[-1, h / 2], [-h * g / (2 * l) * np.cos(yn[0]), -1]])

    for i in range(len(t) - 1):
        def f(yn):
            return (
                np.array([
                    +h / 2 * (y[i, 1] + yn[1]),  # F_1
                    -h * g / (2 * l) * (np.sin(y[i, 0]) + np.sin(yn[0])),  # F_2
                ])
                + y[i, :]  # y_n
                - yn  # y_(n+1)
            )

        y[i + 1, :] = newton(f, y[i, :], jacobian=j)

        if ao is None or i == 0:
            continue  # skip the obstacle checks (no obstacle or not enough data points)

        # create interpolation polynomials of degree 2
        p_a = np.poly1d(np.polyfit(t[i - 1:i + 2], y[i - 1:i + 2, 0], deg=2))
        p_w = np.poly1d(np.polyfit(t[i - 1:i + 2], y[i - 1:i + 2, 1], deg=2))

        # check if any of the roots of p_a-ao are real
        if np.any(np.isreal(roots := (p_a - ao).roots)):
            # check if the roots are in the current time interval [t_i, t_(i+1)]
            in_interval: np.ndarray = (t[i] <= roots) & (roots <= t[i + 1])
            if np.any(in_interval):
                t_obst = roots[in_interval][0]
                # update the next values (at i+1) to be at t_obst
                t[i + 1] = t_obst
                y[i + 1, :] = [ao, p_w(t_obst)]
                # and return all computed steps (discard all other values)
                return t[:i + 2], y[:i + 2, :]

    return t, y


def solve_trapezoidal_with_obstacle(t0, te, y0, h, ao):
    t = [np.array([t0])]
    y = [np.array([y0])]
    while t[-1][-1] < te:
        ti: np.ndarray = t[-1]
        yi: np.ndarray = y[-1]
        tn, yn = solve_trapezoidal_until(
            t0=ti[-1],
            te=te,
            y0=[yi[-1, 0], -yi[-1, 1]],
            h=h,
            ao=ao
        )
        t.append(tn)
        y.append(yn)
    return t, y


# ----------------- plots -----------------


def plot_simple(t0, te, y0, h):
    t, y = solve_trapezoidal_until(t0, te, y0, h)
    a = y[:, 0]
    w = y[:, 1]

    plt.plot(t, a, label="$\\alpha$")
    plt.plot(t, w, label="$\\omega$")
    plt.legend()
    plt.grid()
    plt.show()


def plot_obstacle_simple(t0, te, y0, h):
    ao = -np.pi / 6
    t, y = solve_trapezoidal_with_obstacle(t0, te, y0, h, ao)
    for t, y in zip(t, y):
        plt.plot(t, y[:, 0], color="r")
        plt.plot(t, y[:, 1], color="b")

    plt.axhline(ao, linestyle="--", color="k")
    plt.grid()
    plt.show()


# ------ main -------


def main() -> None:
    y0 = [np.pi/2, 0]
    t0 = 0
    te = 5
    h = 1/50

    # without obstacle
    plot_simple(t0, te, y0, h)

    # TODO: with obstacle
    plot_obstacle_simple(t0, te, y0, h)

if __name__ == '__main__':
    main()


