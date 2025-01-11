from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from verify import pendulum


def newton(function, jacobian, initial_guess, tolerance=1e-5, max_iter=1e3):
    current_guess = np.array(initial_guess, dtype=float)
    for _ in range(int(max_iter)):
        j = jacobian(current_guess)
        f = function(current_guess)
        update = np.linalg.solve(j, f)
        current_guess -= update
        if np.linalg.norm(update) < tolerance:
            break
    return current_guess

class Polynomial:
    """A second order polynomial"""

    def __init__(self, coefficients):
        self.a, self.b, self.c = coefficients

    def __call__(self, x):
        pass

    @staticmethod
    def from_points(points):
        x, y = zip(*points)
        coeff = np.polyfit(x, y, deg=2)
        return Polynomial(coeff)  # FIXME

    @property
    def roots(self):
        if self.a == 0:
            if self.b == 0:
                return np.array([self.c])
            else:
                return np.array([-self.c / self.b])
        else:
            d = np.sqrt(self.b  ** 2 - 4 * self.a * self.c)
            return np.array([
                (-self.b + d) / (2 * self.a),
                (-self.b - d) / (2 * self.a)
            ])

class Pendulum:
    """All pendulum related functions"""

    def __init__(self, a0, w0, t0, te, h, g, l):
        self.a0 = a0
        self.w0 = w0
        self.t0 = t0
        self.te = te
        self.h = h
        self.g = g
        self.l = l
        self.n = int(np.ceil((t0 + te) / h + 1))

    # def _function(self, y, yn) -> np.ndarray:
    #     return np.array([
    #         y[0] + self.h / 2 * (y[1] + yn[1]) - yn[0],
    #         y[1] - self.h * self.g / (2 * self.l) * (np.sin(y[0]) + np.sin(yn[0])) - yn[1]
    #     ])

    def pendulum(self, t, y):
        a, w = y
        return np.array([w, -self.g / self.l * np.sin(a)])

    def _function(self, y, yn) -> np.ndarray:
        return y + self.h/2 * (self.pendulum(0, y) + self.pendulum(0, yn)) - yn
        # return np.array([
        #     y[0] + self.h / 2 * (y[1] + yn[1]) - yn[0],
        #     y[1] - self.h * self.g / (2 * self.l) * (np.sin(y[0]) + np.sin(yn[0])) - yn[1]
        # ])

    def _jacobian(self, yn) -> np.ndarray:
        return np.array([
            [-1,                                              self.h/2],
            [-self.h * self.g / (2 * self.l) * np.cos(yn[0]), -1      ]
        ])

    @staticmethod
    def _polys(t, a, w):  # -> Tuple[Polynomial, Polynomial]:
        # # TODO: compute polynomial coefficients
        # return Polynomial([]), Polynomial([])
        c_a = np.polyfit(t, a, deg=2)
        # print(c_a)
        c_w = np.polyfit(t, w, deg=2)

        return np.poly1d(c_a), np.poly1d(c_w)

    def evaluate(self, y) -> np.ndarray:
        return newton(lambda yn: self._function(y, yn), self._jacobian, y)

    def eval_all(self):
        steps: np.ndarray = np.zeros((self.n, 3))
        steps[0, :] = np.array([self.t0, self.a0, self.w0])

        for i in range(self.n - 1):
            steps[i + 1, 0] = steps[i, 0] + self.h
            steps[i + 1, 1:3] = self.evaluate(steps[i, 1:3])

        return steps

    def _eval_until_obstacle(self, t0, a0, w0, ao: float | int) -> np.ndarray:
        steps: np.ndarray = np.zeros((self.n + 1, 3))
        steps[0, :] = np.array([t0, a0, w0])
        steps[1, :] = np.array([t0 + self.h, *self.evaluate(steps[0, 1:3])])

        i: int = 1
        while steps[i, 0] < self.te:
            steps[i + 1, 0] = steps[i, 0] + self.h
            steps[i + 1, 1:3] = self.evaluate(steps[i, 1:3])

            p_a, p_w = Pendulum._polys(t=steps[i-1:i+2, 0], a=steps[i-1:i+2, 1] - ao, w=steps[i-1:i+2, 2])
            # p_a = Polynomial.from_points(zip(steps[i-1:i+2, 0], steps[i-1:i+2, 1]))
            # p_w = Polynomial.from_points(zip(steps[i-1:i+2, 0], steps[i-1:i+2, 2]))

            # check ip p_a - ao has real roots
            if np.any(np.isreal(roots := p_a.roots)):
                in_interval: np.ndarray = (steps[i, 0] <= roots) & (roots <= steps[i+1, 0])
                if np.any(in_interval):
                    t_obst = roots[in_interval][0]
                    # update the next values (at i+1) to be at t_obst
                    steps[i + 1, :] = np.array([t_obst, ao, p_w(t_obst)])
                    # and return all computed steps (discard all other values)
                    return steps[:i+2, :]
            i += 1
        return steps[:i+1, :]

    def eval_with_obstacle(self, ao: float | int) -> List[np.ndarray]:
        pieces: List[np.ndarray] = [np.array([[self.t0, self.a0, self.w0]])]
        while pieces[-1][-1, 0] < self.te:
            steps: np.ndarray = self._eval_until_obstacle(pieces[-1][-1, 0], pieces[-1][-1, 1], -pieces[-1][-1, 2], ao)
            pieces.append(steps)
        return pieces

# -----------------------------------

def main() -> None:
    pendulum = Pendulum(a0=np.pi/2, w0=0, t0=0, te=5, h=1/500, g=9.81, l=1)

    # without obstacle
    # steps: np.ndarray = pendulum.eval_all()
    # plt.plot(steps[:, 0], steps[:, 1], label="$\\alpha$", color="green")
    # plt.plot(steps[:, 0], steps[:, 2], label="$\\omega$", color="orange")
    m = 1
    l = 1
    g = 9.81
    # with obstacle
    pieces: List[np.ndarray] = pendulum.eval_with_obstacle(-np.pi/6)
    for piece in pieces:
        t = piece[:, 0]
        a = piece[:, 1]
        w = piece[:, 2]
        KE = 0.5 * m * (l * w) ** 2  # Kinetic energy
        PE = m * g * l * (1 - np.cos(a))  # Potential energy
        E_total = KE + PE  # Total energy
        # print(np.max(E_total))

        #plt.plot(t, a, color="r")
        #plt.plot(t, w, color="b")
        plt.plot(t, E_total, label="$\\kappa$", color="blue")

    # plt.axhline(-np.pi / 6, color='red', linestyle='--')

    # show plot
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
