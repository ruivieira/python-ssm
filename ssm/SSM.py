import numpy as np
from numpy import matrix, array
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from scipy.stats import binom


class UnivariateModel:
    def __init__(self):
        pass

    @staticmethod
    def compose(A, B):
        composite = UnivariateModel()
        composite.F = np.hstack((A.F, B.F))
        composite.G = block_diag(*[A.G, B.G])
        composite.m0 = np.hstack((A.m0, B.m0))
        composite.C0 = block_diag(*[A.C0, B.C0])
        composite.W = block_diag(*[A.W, B.W])
        composite.V = A.V + B.V
        return composite


class UnivariateLocallyConstantModel(UnivariateModel):
    def __init__(self, m0=0.0, C0=100.0, W=1.0, V=1.0):
        self.F = matrix([1.0])
        self.G = matrix([1.0])
        self.m0 = array([m0])
        self.C0 = matrix([C0])
        self.W = matrix([W])
        self.V = matrix([V])


class UnivariateFourierModel(UnivariateModel):
    def __init__(self, period, harmonics):
        om = 2.0 * np.pi / period
        harmonic1 = np.identity(2) * np.cos(om)
        harmonic1[0, 1] = np.sin(om)
        harmonic1[1, 0] = -harmonic1[0, 1]

        G = [None] * harmonics
        G[0] = np.copy(harmonic1)
        if harmonics > 1:
            for i in np.arange(1, harmonics):
                G[i] = np.dot(G[i - 1], harmonic1)
            self.G = block_diag(*G)
        else:
            self.G = harmonic1

        self.F = np.matrix([[1.0, 0.0] * harmonics])
        self.m0 = array([0.0] * self.G.shape[0])
        self.C0 = np.diag([100.0] * self.G.shape[0])
        self.W = np.diag([1.0] * self.G.shape[0])
        self.V = matrix([1.0])


def stateGenerator(nobs, model):
    thetas = np.empty((nobs, len(model.m0)))
    theta = multivariate_normal(mean=model.m0, cov=model.C0).rvs()
    for t in np.arange(nobs):
        theta = multivariate_normal(mean=np.dot(model.G, theta),
                                    cov=model.W).rvs()
        thetas[t, :] = theta
    return thetas


class MultivariateModel:
    def __init__(self):
        pass


class MultivariateLocallyConstantModel(MultivariateModel):
    def __init__(self, dimension, m0, C0, W, V):
        self._dimension = dimension
        self.F = np.eye(self._dimension)
        self.G = np.eye(self._dimension)
        self.m0 = m0
        self.C0 = C0
        self.W = W
        self.V = V


def ilogit(alpha):
    return 1.0 / (1.0 + np.exp(-alpha))


def MultivariateBinomialObs(model, trials, states):
    nobs = states.shape[0]
    ys = np.empty((nobs, len(model.m0)))
    for t in np.arange(nobs):
        y = binom(n=trials, p=ilogit(np.dot(model.F, states[t]))).rvs()
        ys[t, :] = y
    return ys
