import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.81
l = 1

def pendulum(t, y):
    a, w = y
    return [w, -g/l * np.sin(a)]

y0 = [np.pi/2, 0]
t0 = 0
te = 5

t = np.linspace(t0, te, 1000)
sol = solve_ivp(pendulum, (t0, te), y0, t_eval=t)

plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[1])
plt.grid(True)
plt.show()
