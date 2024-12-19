from typing import List
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def newton(function, initial_guess, jacobian=None, tolerance=1e-5, max_iter=1e3):
    if jacobian is None:
        jacobian = lambda guess: numerical_jacobian(function, guess)
    current_guess = np.array(initial_guess, dtype=float)

    for _ in range(int(max_iter)):
        # Calculate the Jacobian matrix and function value
        J = jacobian(current_guess)
        F = function(*current_guess)

        # Calculate the update (Jx=F -> x=J^-1F ~> x=F/J)
        update = np.linalg.solve(J, F)

        # Update the guess
        current_guess -= update
        
        # Check for convergence
        if np.linalg.norm(update) < tolerance:
            break

    return current_guess


def numerical_jacobian(func, x, epsilon=1e-6):
    n = len(x)
    m = len(func(*x))
    j = np.zeros((m, n))

    for i in range(n):
        x_shifted = x.copy()
        x_shifted[i] += epsilon

        # assigning the entire column
        j[:, i] = (func(*x_shifted) - func(*x)) / epsilon

    print(j)

    return j

# -----------------------------------------------------------

class Pendulum:
    def __init__(self, a0, w0, t0, te, n, g, l) -> None:
        self.a0 = a0
        self.w0 = w0 
        self.t0 = t0
        self.te = te
        self.n = n
        self.h = (te - t0) / (n - 1)
        self.g = g
        self.l = l
        pass

    def eval(self, t, y) -> np.ndarray:
        F = lambda y1, y2: np.array([
            y[0] + self.h / 2 * (y[1] + y2) - y1,
            y[1] - self.h * self.g / (2 * self.l) * (np.sin(y[0]) + np.sin(y1)) - y2
        ])

        J = lambda yn: np.array([
            [-1, self.h/2],
            [-self.h * self.g / (2 * self.l) * np.cos(yn[0]), -1]
        ])

        return newton(F, y, jacobian=J, tolerance=1e-6, max_iter=1000)

    def func(self, t, y) -> np.ndarray:
        F = lambda y1, y2: np.array([
            y[0] + self.h / 2 * (y[1] + y2) - y1,
            y[1] - self.h * 9.81 / (2 * 1) * (np.sin(y[0]) + np.sin(y1)) - y2
        ])

    def func2(h, t, y) -> np.ndarray:
        F = lambda y1, y2: np.array([
            y[0] + h / 2 * (y[1] + y2) - y1,
            y[1] - h * 9.81 / (2 * 1) * (np.sin(y[0]) + np.sin(y1)) - y2
        ])

    #     return newton(F, y, tolerance=1e-6, max_iter=1000)

    def eval_all(self) -> np.ndarray:
        # every row represents a time step: t, a, w
        steps = np.zeros((self.n, 3))
        steps[:, 0] = np.linspace(self.t0, self.te, self.n)
        steps[0, 1] = self.a0
        steps[0, 2] = self.w0
        for i in range(self.n-1):
            steps[i+1, 1:3] = self.func(steps[i, 0], steps[i, 1:3])
        return steps
    
    @staticmethod
    def polys(t, a, w):
        c_a = np.polyfit(t, a, deg=2)
        c_w = np.polyfit(t, w, deg=2)

        return np.poly1d(c_a), np.poly1d(c_w)


    @staticmethod
    def eval_with_obstacle(a0, w0, t0, te, h, ao: float) -> np.ndarray:
        """
        this function evaluates the angle and the angular velocity
        until the pendulum reaches the obstacle

        returns:
            t_obst: the time when the obstacle is reached (not necessarily a time step)
            a_obst: the angle ...
            w_obst: the velocity ...
            steps: all other intermediary values as a matrix
        """
        n = int((te - t0) / h + 1)
        steps = np.zeros((n, 3))
        steps[:, 0] = np.linspace(t0, te, n)
        steps[0, 1] = a0
        steps[0, 2] = w0
        for i in range(n-1):
            steps[i+1, 1:3] = Pendulum.func2(h, steps[i, 0], steps[i, 1:3])

            if i == 0:
                # not enough data points for interpolation
                continue

            p_a, p_w = Pendulum.polys(steps[i-1:i+2, 0], steps[i-1:i+2, 1], steps[i-1:i+2, 2])

            # check if p_a - ao has real roots (if yes, t a w have to be updated accordingly)
            if np.any(np.isreal(roots := (p_a - ao).r)):
                # check if a root is in the interval [t_i, t_(i+1)]
                in_interval = (steps[i, 0] <= roots) & (roots <= steps[i+1, 0])
                if np.any(in_interval):
                    t_obst = roots[in_interval][0]
                    a_obst = ao  # should be the obstacle angle, otherwise something went completely wrong
                    w_obst = p_w(t_obst)
                    steps[i+1, 0] = t_obst
                    steps[i+1, 1] = a_obst
                    steps[i+1, 2] = w_obst
                    return steps[:i+2, :]

        return steps

def main() -> None:
    pendulum = Pendulum(np.pi / 2, 0, 0, 5, 500, 9.81, 1)
    vals = pendulum.eval_all()
    plt.plot(vals[:, 0], vals[:, 1], label="$\\alpha$")
    plt.plot(vals[:, 0], vals[:, 2], label="$\\omega$")

    print(np.min(vals[:, 2]), np.max(vals[:, 2]))

    # te = 5
    # h = 1/100
    # steps = np.array([[0, np.pi/2, 0]])
    # while steps[-1, 0] < te:
    #     steps =  pendulum.eval_with_obstacle(steps[-1, 1], -steps[-1, 2], steps[-1, 0], te, h, -np.pi/6)
    #     plt.plot(steps[:, 0], steps[:, 1], label="$\\alpha_{obst}=-\pi/6$")
    # plt.axhline(-np.pi/6, color='red', linestyle='--')
    
    plt.legend()
    plt.xlabel("time t")
    plt.ylabel("angle in radians & angular velocity")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
