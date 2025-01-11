import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
l = 1        # Length of the pendulum (m)
g = 9.81     # Gravitational acceleration (m/s^2)
m = 1        # Mass of the pendulum (kg)
theta0 = np.pi / 4  # Initial angle (rad)
omega0 = 0          # Initial angular velocity (rad/s)

# Time range
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Define the equations of motion
def pendulum_ode(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Solve the differential equation
solution = solve_ivp(pendulum_ode, t_span, [theta0, omega0], t_eval=t_eval)

# Extract theta and omega
theta = solution.y[0]
omega = solution.y[1]

# Calculate energies
KE = 0.5 * m * (l * omega)**2  # Kinetic energy
PE = m * g * l * (1 - np.cos(theta))  # Potential energy
E_total = KE + PE  # Total energy

# Plot energies
plt.figure(figsize=(10, 6))
plt.plot(t_eval, KE, label="Kinetic Energy (KE)", color='blue')
plt.plot(t_eval, PE, label="Potential Energy (PE)", color='orange')
plt.plot(t_eval, E_total, label="Total Energy (E)", color='green', linestyle='dashed')
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Energy of a Pendulum Over Time")
plt.legend()
plt.grid()
plt.show()