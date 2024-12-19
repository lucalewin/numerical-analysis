import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def cubspline(xint, yint):
    """
    compute the cubic spline coefficients for the given nodes

    params:
        xint: equidistant (!) x values
        yint: the function values at the corresponding x values
    """
    m = len(xint) - 1
    h = (xint[-1] - xint[0]) / m

    A = np.eye(m-1) * 4 + np.eye(m-1, k=-1) + np.eye(m-1, k=1)
    b = (6 / h**2) * (yint[2:m+1] - 2*yint[1:m] + yint[0:m-1])

    sigma = np.linalg.solve(A, b)
    sigma = np.concat([[0], sigma, [0]])  # add the boundary conditions

    a = (sigma[1:] - sigma[:-1]) / (6 * h)
    b = 0.5 * sigma[:-1]
    c = (yint[1:] - yint[:-1]) / h - h * (sigma[1:] + 2 * sigma[:-1]) / 6
    d = yint[:-1]

    return np.column_stack((a, b, c, d))

def cubsplineval(coeff, xint, xval):
    """
    Compute the value of the spline for a specific x-value

    params:
        coeff: The cubic spline coefficients
        xint: The x-values of the interpolation nodes
        xval: The x-value to evaluate
    """
    for i in range(len(xint) - 1):
        # equality can be use here on both ends because the splines have the
        # same y-value at the points where they meet
        if xint[i] <= xval <= xint[i+1]:
            dx = xval - xint[i]
            return coeff[i,0] * dx**3 + coeff[i,1] * dx**2 + coeff[i,2] * dx + coeff[i,3]
    raise ValueError(f"Outside of the interpolation interval [{min(xint)}, {max(xint)}], {xval}")

def case1():
    f = lambda x: np.exp(-4 * x**2)
    x = np.linspace(-1, 1, 15, endpoint=True)
    y = np.array(f(x))

    coeff = cubspline(x, y)

    x_full = np.linspace(-1, 1, 250)
    y_full = f(x_full)
    y_interp = [cubsplineval(coeff, x, x_i) for x_i in x_full]

    plt.scatter(x, y, label="nodes")
    plt.plot(x_full, y_full, label="actual function")
    plt.plot(x_full, y_interp, label="cubic splines")
    plt.title("Cubic Spline Interpolation of $f(x)= exp(-4x^2)$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

def case2():
    f = lambda x: 1 / (1 + 25 * x**2)
    x = np.linspace(-1, 1, 15, endpoint=True)
    y = np.array(f(x))

    coeff = cubspline(x, y)

    x_full = np.linspace(-1, 1, 250)
    y_full = f(x_full)
    y_interp = [cubsplineval(coeff, x, x_i) for x_i in x_full]

    plt.scatter(x, y, label="nodes")
    plt.plot(x_full, y_full, label="actual function")
    plt.plot(x_full, y_interp, label="cubic splines")
    plt.title("Cubic Spline Interpolation of $f(x)= 1 / (1+25x^2)$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

def problem3():
    from s1002_fixed import s1002

    x = np.linspace(-69, 60, 250)
    y = s1002(x)

    x1 = np.linspace(-69, 60, 15)
    y1 = s1002(x1)

    coeff = cubspline(x1, y1)
    cy = [cubsplineval(coeff, x1, x_i) for x_i in x]

    plt.scatter(x1, y1, label="nodes")
    plt.plot(x, cy, label="spline")
    plt.plot(x, y, label="actual profile")
    plt.title("The S1002 wheel profile and its cubic splines interpolation")
    plt.xlabel("$x$ [mm]")
    plt.ylabel("$y$ [mm]")
    plt.legend()
    plt.show()

# case1()
# case2()
problem3()
