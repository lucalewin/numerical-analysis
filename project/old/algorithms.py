import numpy as np


def newton(function, initial_guess, jacobian=None, tolerance=1e-5, max_iter=1e3):
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
