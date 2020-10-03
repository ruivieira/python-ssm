"""Monte Carlo methods"""
# pylint: disable=no-else-return,too-many-locals,too-many-arguments
import numpy as np
import numpy.linalg as linalg
import numpy.random as rand

from pssm.filters import KalmanFilter


class FFBS:
    """Forward filtering, backward sampling"""

    def __init__(self, structure, m0, C0, vprior, wprior, data):
        self._structure = structure
        self._m0 = m0
        self._C0 = C0
        self._vprior = vprior
        self._wprior = wprior
        self._data = data
        self._nobs = len(data)

    @staticmethod
    def backward_sampling_step(theta, m, C, W):
        """Backward sampling step"""
        _inv = linalg.inv(C + W)
        mean = (
            m
            + np.asarray(np.dot(C, np.asarray(np.dot(_inv, theta - m)).ravel())).ravel()
        )
        cov = C - C * np.transpose(C * _inv)

        return rand.multivariate_normal(mean=mean, cov=cov)

    @staticmethod
    def backward_sampling(ms, Cs, W):
        """Backward sampling"""
        T = len(ms) - 1
        theta = rand.multivariate_normal(
            mean=np.atleast_1d(ms[T]), cov=np.atleast_2d(Cs[T])
        )
        thetas = [theta]
        for t in reversed(range(1, T)):
            theta = FFBS.backward_sampling_step(theta, ms[t], Cs[t], W)
            thetas.append(theta)
        return list(reversed(thetas))

    def _states(self, V, W):
        ms = [self._m0]
        Cs = [self._C0]
        structure = self._structure
        structure._W = W  # pylint: disable=protected-access
        kalman = KalmanFilter(structure=structure, V=V)
        for t in range(1, self._nobs):
            m, C = kalman.filter(self._data[t], ms[t - 1], Cs[t - 1])
            ms.append(m)
            Cs.append(C)
        return ms, Cs

    def run(self, V, W, states=False):
        """Run the FFBS algorithm"""
        Ft = self._structure.F.T
        n = self._nobs - 1

        ms, Cs = self._states(V=V, W=W)

        thetas = FFBS.backward_sampling(ms, Cs, W)

        ssy = np.zeros(n)
        sstheta = np.zeros((n, len(self._wprior)))
        for t in range(1, n):
            ssy[t] = np.power(self._data[t] - np.dot(Ft, thetas[t]), 2.0)
            sstheta[t] = np.power(
                thetas[t] - np.dot(self._structure.G, thetas[t - 1]), 2.0
            )

        v_rate = self._vprior + 0.5 * sum(ssy)
        _V = 1.0 / rand.gamma(self._vprior + 0.5 * n, scale=v_rate)

        w_rate = [self._vprior + 0.5 * d for d in sstheta.sum(axis=0)]
        _W = np.diag(
            [
                1.0 / rand.gamma(prior + 0.5 * n, scale=scale)
                for scale, prior in zip(w_rate, self._wprior)
            ]
        )

        if states:
            return _W, _V, thetas
        else:
            return _W, _V
