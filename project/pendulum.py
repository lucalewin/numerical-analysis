from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt


g = 9.81  # m/s^2
l = 1  # m
m = 1  # kg


def newton(
    function: Callable,
    initial_guess: List | np.ndarray,
    jacobian: Callable = None,
    tolerance: float | int = 1e-5,
    max_iter: int = 1000,
):
    if jacobian is None:
        jacobian = lambda guess: numerical_jacobian(function, guess)
    current_guess = np.array(initial_guess, dtype=float)
    for _ in range(int(max_iter)):
        j = jacobian(current_guess)
        f = function(current_guess)
        update = np.linalg.solve(j, f)
        current_guess -= update
        if np.linalg.norm(update) < tolerance:
            break
    return current_guess


def numerical_jacobian(func, x, epsilon=1e-6):
    n = len(x)
    m = len(func(x))
    j = np.zeros((m, n))
    for i in range(n):
        x_shifted = x.copy()
        x_shifted[i] += epsilon
        j[:, i] = (func(x_shifted) - func(x)) / epsilon
    return j


def pendulum(_, y):
    """Compute the two first-order ODEs of a pendulum."""
    return np.array([y[1], -g / l * np.sin(y[0])])


def solve_trapezoidal(func, t0, te, y0, n):
    """General implementation of the trapezoidal rule for solving initial value problems"""
    t = np.linspace(t0, te, num=n)
    y = np.zeros((n, 2))
    h = (te - t0) / (n - 1)  # Step size
    y[0, :] = y0

    for i in range(n - 1):
        # This function is the actual trapezoidal rule
        def f(yn):
            return y[i] + h / 2 * (func(t[i], y[i]) + func(t[i + 1], yn)) - yn

        # Use the Newton's iteration to find a root of F
        # (using a numerical method to compute the Jacobian matrix)
        y[i + 1, :] = newton(f, y[i, :])

    return t, y


def solve_trapezoidal_pendulum(
    t0: int | float, te: int | float, y0: int | float | List | np.ndarray, n: int
):
    """A specific implementation of the trapezoidal algorithm for a pendulum."""
    t = np.linspace(t0, te, num=n)
    y = np.zeros((n, 2))
    h = (te - t0) / (n - 1)  # Step size
    y[0, :] = y0

    # define the jacobian matrix function
    def j(yn):
        return np.array([[-1, h / 2], [-h * g / (2 * l) * np.cos(yn[0]), -1]])

    for i in range(n - 1):
        # define the function F to solve F=0
        def f(yn):
            return (
                np.array(
                    [
                        +h / 2 * (y[i, 1] + yn[1]),
                        -h * g / (2 * l) * (np.sin(y[i, 0]) + np.sin(yn[0])),
                    ]
                )
                + y[i, :]
                - yn
            )

        # Use the Newton's iteration to approximate the
        # root of F (using the exact Jacobian matrix)
        y[i + 1, :] = newton(f, y[i, :], jacobian=j)

    return t, y
