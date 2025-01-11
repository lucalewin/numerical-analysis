import numpy as np
import matplotlib.pyplot as plt

# Define a convex function
def convex_f(x):
    # return x**2.8+0.3
    return -(x-0)**2 + 1

# Points for trapezoidal segment
x0, x1 = 0, 1  # Interval for the trapezoid
a, b = 0.2, 0.8
y0, y1 = convex_f(a), convex_f(b)

# Generate points for the curve
x_curve = np.linspace(x0, x1, 500)
y_curve = convex_f(x_curve)

# Points for the trapezoid
x_trap = [a, b, b, a]
y_trap = [0, 0, y1, y0]

# Prepare the plot
plt.figure(figsize=(4, 5))

# Plot the convex curve
plt.plot(x_curve, y_curve, label="$g'(x)$", color="black")

# Plot and fill the trapezoid below the curve
plt.fill(x_trap, y_trap, color="red", alpha=0.3)

# Mark the points of the trapezoid
plt.scatter([a, b], [y0, y1], color="orange", marker="x", zorder=5)

# Add labels, legend, and grid
# plt.title("Trapezoidal Rule with a Convex Curve")
# plt.xlabel("x")
# plt.ylabel("f(x)")
plt.legend()
# plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()