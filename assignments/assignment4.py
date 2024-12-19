import numpy as np
import matplotlib.pyplot as plt


def adaptive_simpson(f, a, b, eps=5e-6):
    intervals = []

    def simpson(f, a, b):
        """Compute the Simpson's rule for the interval [a, b]."""
        return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))
    
    def recursive(f, a, b, eps):
        """Recursive function for adaptive Simpson's rule."""
        c = (a + b) / 2
        s1 = simpson(f, a, b)
        s2 = simpson(f, a, c) + simpson(f, c, b)

        # add the current interval
        intervals.append((a, b))

        # check tolerance
        if abs(s2 - s1) <= eps * 15:
            return s2 + (s2 - s1) / 15
        else:
            l = recursive(f, a, c, eps/2)
            r = recursive(f, c, b, eps/2)
            return l + r

    result = recursive(f, a, b, eps)
    return result, intervals


def main():
    f = lambda x: np.sqrt(np.abs(x))
    a, b = -1, 1
    epsilon = 5e-6

    # Compute the integral and record intervals
    result, intervals = adaptive_simpson(f, a, b, epsilon)

    # Plot the adaptive subintervals
    x = np.linspace(a, b, 500)
    y = f(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='f(x) = $\\sqrt{|x|}$', color='blue')

    # plot vertical lines indicating the intervals
    for interval in intervals:
        plt.axvline(interval[0], color='red', linestyle='--', alpha=0.7)
    plt.axvline(intervals[-1][1], color='red', linestyle='--', alpha=0.7)

    plt.title('Adaptive Simpson\'s Rule (with subintervals)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Computed integral = {result}")


if __name__ == "__main__":
    main()
