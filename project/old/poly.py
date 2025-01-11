import numpy as np

class Polynomial:
    """Second degree polynomial"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def from_points(x, y):
        pass

    @property
    def roots(self) -> np.ndarray:
        return np.array([])

    def __call__(self, x):
        return self.a * x**2 + self.b * x + self.c
