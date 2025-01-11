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
    current_guess = np.array(initial_guess, dtype=float)
    for _ in range(int(max_iter)):
        j = jacobian(current_guess)
        f = function(current_guess)
        update = np.linalg.solve(j, -f)
        current_guess += update
        if np.linalg.norm(update) < tolerance:
            return current_guess
    raise Exception("newtons iteration did not converge")


class Polynomial:
    def __init__(self, coefficients):
        self.a, self.b, self.c = coefficients

    @staticmethod
    def fit(x, y):
        if len(x) != 3 or len(y) != 3:
            raise ValueError("Exactly three points are required for quadratic interpolation.")
        A = np.array([
            [x[0]**2, x[0], 1],
            [x[1]**2, x[1], 1],
            [x[2]**2, x[2], 1],
        ])
        return np.linalg.solve(A, y)

    def __add__(self, constant):
        return Polynomial([self.a, self.b, self.c + constant])

    def __sub__(self, constant):
        return Polynomial([self.a, self.b, self.c - constant])

    def __call__(self, x):
        return self.a * x**2 + self.b * x + self.c

    def roots(self):
        """compute roots of quadratic polynomial"""
        a, b, c = self.a, self.b, self.c
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            root1 = (-b - np.sqrt(discriminant)) / (2 * a)
            root2 = (-b + np.sqrt(discriminant)) / (2 * a)
        else:
            import cmath  # use to compute complex roots
            root1 = (-b - cmath.sqrt(discriminant)) / (2 * a)
            root2 = (-b + cmath.sqrt(discriminant)) / (2 * a)
        return np.array([root1, root2])


def interpolate(t, y):
    return (
        Polynomial(Polynomial.fit(t, y[:, 0])),
        Polynomial(Polynomial.fit(t, y[:, 1]))
    )


def solve_trapezoidal(
        t0: float | int,
        te: float | int,
        y0: int | float | List | np.ndarray,
        h: float | int,
        ao: float | int = None
):
    t = np.arange(t0, te + h, step=h)
    y = np.zeros((len(t), 2))
    y[0, :] = y0  # set the initial conditions

    # define the Jacobian matrix function J_F
    def j(yn):
        return np.array([[-1, h / 2], [-h * g / (2 * l) * np.cos(yn[0]), -1]])

    for i in range(len(t) - 1):
        # define the function F
        def f(yn):
            return (
                np.array([
                    +h / 2 * (y[i, 1] + yn[1]),  # F_1 (partly)
                    -h * g / (2 * l) * (np.sin(y[i, 0]) + np.sin(yn[0])),  # F_2 (partly)
                ])
                + y[i, :]  # y_n
                - yn  # y_(n+1)
            )

        # iteratively solve the trapezoidal system
        y[i + 1, :] = newton(f, y[i, :], jacobian=j)

        if ao is None or i == 0:
            continue  # skip the obstacle checks (no obstacle or not enough data points)

        # create interpolation polynomials
        p_a, p_w = interpolate(t[i - 1:i + 2], y[i - 1:i + 2, :])

        # check if any of the roots of p_a-ao are real
        if np.any(np.isreal(roots := (p_a - ao).roots())):
            # check if the roots are in the current time interval [t_i, t_(i+1)]
            in_interval: np.ndarray = (t[i] <= roots) & (roots <= t[i + 1])
            if np.any(in_interval):
                t_obst = roots[in_interval][0]
                # update the next values (at t_(i+1)) to be at t_obst
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
        tn, yn = solve_trapezoidal(
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
    t, y = solve_trapezoidal(t0, te, y0, h)
    a = y[:, 0]
    w = y[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot alpha in the first subplot
    axes[0].plot(t, a, label="$\\alpha$", color="blue")
    axes[0].legend()
    axes[0].grid()
    axes[0].set_ylabel("Angle [rad]")

    # Plot omega in the second subplot
    axes[1].plot(t, w, label="$\\omega$", color="red")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Angular velocity [rad/s]")

    fig.suptitle("Trajectory and Velocity of the Pendulum")
    plt.tight_layout()
    plt.show()


def plot_different_step_size(t0, te, y0, h):
    for i in range(len(h)):
        t, y = solve_trapezoidal(t0, te, y0, h[i])
        plt.plot(t, y[:, 0], label=f"$\\alpha$ (h={h[i]})")
    plt.legend()
    plt.title("Trajectory of the Pendulum for different step sizes")
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.grid()
    plt.show()


def plot_energy(t0, te, y0, h):
    plt.figure(figsize=(10, 6))
    for hi in h:
        t, y = solve_trapezoidal(t0, te, y0, hi)
        a = y[:, 0]
        w = y[:, 1]
        KE = 0.5 * m * (l * w) ** 2  # Kinetic energy
        PE = m * g * l * (1 - np.cos(a))  # Potential energy
        E_total = KE + PE
        plt.plot(t, E_total, label=f"Total Energy (E) h={hi}")

    plt.legend()
    plt.title("Energy of Pendulum for different step sizes")
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.grid()
    plt.show()


def plot_energy_error(t0, te, y0):
    plt.figure(figsize=(10, 6))

    h = np.logspace(-4, 0, 20)
    error = []

    for hi in h:
        t, y = solve_trapezoidal(t0, te, y0, hi)
        a = y[:, 0]
        w = y[:, 1]

        KE = 0.5 * m * (l * w) ** 2  # Kinetic energy
        PE = m * g * l * (1 - np.cos(a))  # Potential energy
        E_total = KE + PE

        # Compute the error as the maximum deviation from initial total energy
        e = np.abs(E_total - E_total[0])
        error.append(np.max(e))

    plt.plot(h, error, marker='o', label="Energy Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Step size (h)")
    plt.ylabel("Maximum Energy Error")
    plt.title("Energy Error vs. Step Size")
    plt.legend()
    plt.grid()
    plt.show()


def plot_phase_space(t0, te, y0, h):
    _, y = solve_trapezoidal(t0, te, y0, h)
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
    plt.plot(a, w, label=f"Phase plot (h={h})", color="red")
    plt.streamplot(A, W, dA, dW, color='#1f77b4')
    plt.legend()
    plt.title("Phase space of the Pendulum")
    plt.xlabel('Angle [rad]')
    plt.ylabel('Angular velocity [rad/s]')
    plt.grid()
    plt.show()


# ------------------- obstacle plots ------------------------


def plot_obstacle_simple(t0, te, y0, h, ao):
    t, y = solve_trapezoidal_with_obstacle(t0, te, y0, h, ao)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot y[:, 0] in the first subplot
    for i, (t_piece, y_piece) in enumerate(zip(t, y)):
        axes[0].plot(t_piece, y_piece[:, 0], color="r", label=i*"_"+f"$\\alpha$")
    axes[0].axhline(ao, linestyle="--", color="k", label="$\\alpha_{obst}$")
    axes[0].grid()
    axes[0].set_ylabel("Angle [rad]")
    axes[0].legend()

    # Plot y[:, 1] in the second subplot
    for i, (t_piece, y_piece) in enumerate(zip(t, y)):
        axes[1].plot(t_piece, y_piece[:, 1], color="b", label=i*"_"+f"$\\omega$")
    axes[1].grid()
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Angular velocity [rad/s]")
    axes[1].legend()

    fig.suptitle("Trajectory and Velocity of the Pendulum")
    plt.tight_layout()
    plt.show()


def plot_obstacle_different_step_size(t0, te, y0, h, ao):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # from matplotlib
    for i in range(len(h)):
        t, y = solve_trapezoidal_with_obstacle(t0, te, y0, h[i], ao=ao)
        for j, (t, y) in enumerate(zip(t, y)):
            plt.plot(t, y[:, 0], color=colors[i], label=j*"_"+f"h={h[i]}")

    plt.legend()
    plt.axhline(ao, linestyle="--", color="k")
    plt.xlabel('time')
    plt.ylabel('angle')
    plt.grid()
    plt.show()


def plot_obstacle_energy(t0, te, y0, h, ao):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # from matplotlib
    for i, hi in enumerate(h):
        t, y = solve_trapezoidal_with_obstacle(t0, te, y0, hi, ao)
        for j, (t, y) in enumerate(zip(t, y)):
            a = y[:, 0]
            w = y[:, 1]
            KE = 0.5 * m * (l * w) ** 2  # Kinetic energy
            PE = m * g * l * (1 - np.cos(a))  # Potential energy
            E_total = KE + PE
            plt.plot(t, E_total, color=colors[i], label=j*"_"+f"h={hi}")

    plt.legend()
    plt.title("Energy of Pendulum (with Obstacle) for different step sizes")
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.grid()
    plt.show()


def plot_obstacle_phase_space(t0, te, y0, h, ao):
    _, y = solve_trapezoidal_with_obstacle(t0, te, y0, h, ao=ao)
    y2 = np.concat(y)
    a = y2[:, 0]
    w = y2[:, 1]

    num = 33
    full_a = np.linspace(np.min(a) - 1, np.max(a) + 1, num)
    full_w = np.linspace(np.min(w) - 4, np.max(w) + 4, num)

    A, W = np.meshgrid(full_a, full_w)

    # partial derivatives at every point
    dA = W
    dW = -g / l * np.sin(A)

    plt.figure(figsize=(8, 8))

    for i, y in enumerate(y):
        a = y[:, 0]
        w = y[:, 1]
        plt.plot(a, w, label=i*"_"+f"Phase plot (h={h})", color='red', linewidth=2)
    plt.streamplot(A, W, dA, dW, color='#1f77b4')
    plt.legend()
    plt.title("Phase space of Pendulum (with Obstacle)")
    plt.xlabel('Angle [rad]')
    plt.ylabel('Angular velocity [rad/s]')
    plt.grid()
    plt.show()


# ------ main -------


def main() -> None:
    y0 = [np.pi/2, 0]
    t0 = 0
    te = 5
    h = 1/500
    ao = -np.pi / 6

    # without obstacle
    plot_simple(t0, te, y0, h)
    plot_different_step_size(t0, te, y0, [1/2, 1/5, 1/10, 1/100])
    plot_energy(t0, te, y0, [1/50, 1/100, 1/200, 1/1000])
    plot_energy_error(t0, te, y0)
    plot_phase_space(t0, te, y0, h)


    # with obstacle
    plot_obstacle_simple(t0, te, y0, h, ao)
    ## plot_obstacle_different_step_size(t0, te, y0, [1/2, 1/5, 1/10, 1/100], ao)
    plot_obstacle_energy(t0, te, y0, [1/50, 1/100, 1/200, 1/1000], ao)
    plot_obstacle_phase_space(t0, te, y0, h, ao)


if __name__ == '__main__':
    main()
