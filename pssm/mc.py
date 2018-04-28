import numpy as np
from numpy.linalg import inv
from scipy.stats import invgamma
from scipy.stats import multivariate_normal as mvn

from pssm.filters import KalmanFilter


class FFBS:
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
        _inv = inv(C + W)
        mean = m + np.dot(C, np.dot(_inv, theta - m).A1).A1
        cov = C - C * np.transpose(C * _inv)

        return mvn(mean=mean, cov=cov).rvs()

    @staticmethod
    def backward_sampling(ms, Cs, W):
        T = len(ms) - 1
        theta = mvn(mean=np.asarray(ms[T]), cov=Cs[T]).rvs()
        thetas = [theta]
        for t in reversed(range(1, T)):
            theta = FFBS.backward_sampling_step(theta, ms[t], Cs[t], W)
            thetas.append(theta)
        return list(reversed(thetas))

    def _states(self, V, W):
        ms = [self._m0]
        Cs = [self._C0]
        structure = self._structure
        structure._W = W
        kf = KalmanFilter(structure=structure, V=V)
        for t in range(1, self._nobs):
            m, C = kf.filter(self._data[t], ms[t - 1], Cs[t - 1])
            ms.append(m)
            Cs.append(C)
        return ms, Cs

    def run(self, V, W, states=False):
        Ft = self._structure.F.T
        n = self._nobs - 1

        ms, Cs = self._states(V=V, W=W)

        thetas = FFBS.backward_sampling(ms, Cs, W)

        ssy = np.zeros(n)
        sstheta = np.zeros((n, len(self._wprior)))
        for t in range(1, n):
            ssy[t] = np.power(self._data[t] - np.dot(Ft, thetas[t]), 2.0)
            sstheta[t] = np.power(
                thetas[t] - np.dot(self._structure.G, thetas[t - 1]), 2.0)

        v_rate = self._vprior + 0.5 * sum(ssy)
        _V = invgamma(self._vprior + 0.5 * n, scale=v_rate).rvs()

        w_rate = [self._vprior + 0.5 * d for d in sstheta.sum(axis=0)]
        _W = np.diag(
            [invgamma(prior + 0.5 * n, scale=scale).rvs()
             for scale, prior in zip(w_rate, self._wprior)])

        if states:
            return _W, _V, thetas
        else:
            return _W, _V
