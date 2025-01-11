import numpy as np
import matplotlib.pyplot as plt

g = 9.81
l = 1

# num = 100
# a = np.linspace(-np.pi, np.pi, num)
# w = np.linspace(-3 * np.pi, 3 * np.pi, num)
#
# A, W = np.meshgrid(a, w)
#
# # derivatives
# dA = W
# dW = -g/l * np.sin(A)
#
# # magnitude
# m = np.sqrt(dA**2 + dW**2)
#
# plt.figure(figsize=(8, 8))
# quiver = plt.quiver(A, W, dA / m, dW / m, m, cmap='plasma', pivot='mid', scale=num)
# cbar = plt.colorbar(quiver)
# cbar.set_label('Vector Magnitude')
# plt.grid()
# plt.show()