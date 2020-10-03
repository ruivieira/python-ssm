"""Module defining SSM constructs"""
# pylint: disable=too-many-arguments,too-few-public-methods,invalid-name
from __future__ import annotations
import numpy as np  # type: ignore
from numpy import matrix, array  # type: ignore
from scipy.linalg import block_diag  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore
from scipy.stats import binom  # type: ignore


class UnivariateModel:
    """Defines a SSM univariate model"""

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        m0: np.ndarray,
        C0: np.ndarray,
        W: np.ndarray,
        V: float,
    ):
        self.F = F
        self.G = G
        self.m0 = m0
        self.C0 = C0
        self.W = W
        self.V = V

    @staticmethod
    def compose(model_a: UnivariateModel, model_b: UnivariateModel) -> UnivariateModel:
        """Creates a new model composition from two univariate models"""
        composite = UnivariateModel(
            F=np.hstack((model_a.F, model_b.F)),
            G=block_diag(*[model_a.G, model_b.G]),
            m0=np.hstack((model_a.m0, model_b.m0)),
            C0=block_diag(*[model_a.C0, model_b.C0]),
            W=block_diag(*[model_a.W, model_b.W]),
            V=model_a.V + model_b.V,
        )
        return composite


class UnivariateLocallyConstantModel(UnivariateModel):
    """Represents a univariate locally constant model"""

    def __init__(self, m0=0.0, C0=100.0, W=1.0, V=1.0):
        super().__init__(
            F=matrix([1.0]),
            G=matrix([1.0]),
            m0=array([m0]),
            C0=matrix([C0]),
            W=matrix([W]),
            V=matrix([V]),
        )


class UnivariateFourierModel(UnivariateModel):
    """Represents a univariate seasonal (Fourier) model"""

    def __init__(self, period, harmonics):
        omega = 2.0 * np.pi / period
        harmonic1 = np.identity(2) * np.cos(omega)
        harmonic1[0, 1] = np.sin(omega)
        harmonic1[1, 0] = -harmonic1[0, 1]

        G = [None] * harmonics
        G[0] = np.copy(harmonic1)
        if harmonics > 1:
            for i in np.arange(1, harmonics):
                G[i] = np.dot(G[i - 1], harmonic1)
            G = block_diag(*G)
        else:
            G = harmonic1

        super().__init__(
            F=np.matrix([[1.0, 0.0] * harmonics]),
            G=G,
            m0=array([0.0] * G.shape[0]),
            C0=np.diag([100.0] * G.shape[0]),
            W=np.diag([1.0] * G.shape[0]),
            V=matrix([1.0]),
        )


def state_generator(nobs, model):
    """Generate nobs observations for a model"""
    thetas = np.empty((nobs, len(model.m0)))
    theta = multivariate_normal(mean=model.m0, cov=model.C0).rvs()
    for t in np.arange(nobs):
        theta = multivariate_normal(mean=np.dot(model.G, theta), cov=model.W).rvs()
        thetas[t, :] = theta
    return thetas


class MultivariateModel:
    """Generic multivariate model class"""

    def __init__(
        self,
        F: np.matrix,
        G: np.matrix,
        m0: np.matrix,
        C0: np.matrix,
        W: np.matrix,
        V: np.matrix,
    ):
        self.F = F
        self.G = G
        self.m0 = m0
        self.C0 = C0
        self.W = W
        self.V = V


class MultivariateLocallyConstantModel(MultivariateModel):
    """Represents a locally constant multivariate model"""

    def __init__(self, dimension, m0, C0, W, V):
        self._dimension = dimension
        super().__init__(
            F=np.eye(self._dimension), G=np.eye(self._dimension), m0=m0, C0=C0, W=W, V=V
        )


def ilogit(alpha):
    """Inverse-logit transformation"""
    return 1.0 / (1.0 + np.exp(-alpha))


def multivariate_binomial_obs(model, trials, states):
    """Generate observations for a multivariate binomial model"""
    nobs = states.shape[0]
    ys = np.empty((nobs, len(model.m0)))
    for t in np.arange(nobs):
        y = binom(n=trials, p=ilogit(np.dot(model.F, states[t]))).rvs()
        ys[t, :] = y
    return ys
