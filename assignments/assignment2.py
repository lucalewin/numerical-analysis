import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

def case1():
    f = lambda x: np.exp(-4 * x**2)

    x1 = np.linspace(-1, 1, 5, endpoint=True)
    y1 = f(x1)

    x2 = np.linspace(-1, 1, 12, endpoint=True)
    y2 = f(x2)

    x_full = np.linspace(-1, 1, 100, endpoint=True)
    y_full = f(x_full)

    i1 = lagrange(x1, y1)
    i2 = lagrange(x2, y2)

    iy1 = i1(x_full)
    iy2 = i2(x_full)

    plt.scatter(x1, y1, marker='.', label="5 nodes")
    plt.scatter(x2, y2, marker='.', label="12 nodes")
    plt.plot(x_full, iy1, label="interpolation with 5 nodes")
    plt.plot(x_full, iy2, label="interpolation with 12 nodes")
    plt.plot(x_full, y_full, label="actual function")
    plt.title("Interpolating $f(x)= exp(-4x^2)$ in the Lagrange basis")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

def case2():
    f = lambda x: 1 / (1 + 25 * x**2)

    x1 = np.linspace(-1, 1, 5, endpoint=True)
    y1 = f(x1)

    x2 = np.linspace(-1, 1, 12, endpoint=True)
    y2 = f(x2)

    x_full = np.linspace(-1, 1, 100, endpoint=True)
    y_full = f(x_full)

    i1 = lagrange(x1, y1)
    i2 = lagrange(x2, y2)

    iy1 = i1(x_full)
    iy2 = i2(x_full)

    print(i1)

    print(min(iy1), min(iy2))

    plt.scatter(x1, y1, marker='.', label="15 nodes")
    plt.scatter(x2, y2, marker='.', label="21 nodes")
    plt.plot(x_full, iy1, label="interpolation with 15 nodes")
    plt.plot(x_full, iy2, label="interpolation with 21 nodes")
    plt.plot(x_full, y_full, label="actual function")
    plt.title("Interpolating $f(x)= 1 / (1+25x^2)$ in the Lagrange basis")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # plt.ylim((-5, 8))
    plt.legend()
    plt.show()

# case1()
case2()