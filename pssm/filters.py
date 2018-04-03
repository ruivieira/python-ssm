import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, structure, V):
        self._structure = structure
        self._V = V
        self._Ft = np.transpose(self._structure.F)

    def _prior_state(self, m):
        a = np.dot(self._structure.G, m).A1
        return a

    def _prior_covariance(self, C):
        R = self._structure.G * C * self._structure.G.T + self._structure.W
        return R

    def _innovation(self, a, y):
        e = y - np.asscalar(self._Ft.dot(a))
        return e

    def _residual_covariance(self, R):
        Q = self._Ft * R * self._structure.F + self._V
        return Q

    def _gain(self, R, Q):
        K = R * self._structure.F * inv(Q)
        return K

    def filter(self, y, m, C):
        # Predicted (a priori) state estimate
        a = self._prior_state(m)

        # Predicted (a priori) estimate covariance
        R = self._prior_covariance(C)

        # Innovation or measurement residual
        e = self._innovation(a, y)

        # Innovation (or residual) covariance
        Q = self._residual_covariance(R)

        # Kalman gain
        K = self._gain(R, Q)

        new_m = a + K.A1 * e
        new_C = R - K * Q * K.T
        return new_m, new_C
