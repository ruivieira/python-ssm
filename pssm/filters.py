import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, structure, V):
        self._structure = structure
        self._V = V
        self._Ft = self._structure.F.T

    def filter(self, y, m, C):
        # Predicted (a priori) state estimate
        a = np.dot(self._structure.G, m).A1

        # Predicted (a priori) estimate covariance
        R = self._structure.G * C * self._structure.G.T + self._structure.W

        f = np.asscalar(np.dot(self._structure.F, a))

        # Innovation or measurement residual
        e = y - f

        # Innovation (or residual) covariance
        Q = self._structure.F * R * self._Ft + self._V

        # Kalman gain
        K = R * self._Ft * inv(Q)

        new_m = a + K.A1 * e
        new_C = R - K * Q * K.T
        return new_m, new_C
