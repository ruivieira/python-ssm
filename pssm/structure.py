"""Module defining DGLM structures"""
from __future__ import annotations
from typing import List
from math import cos, sin
import numpy as np
from pssm.utils import block_diag


class DimensionError(Exception):
    """Dimension error exception"""


class UnivariateStructure:
    """A univariate structure"""

    def __init__(self, F, G, W):
        self._F = F
        self._G = G
        self._W = W

    @property
    def F(self):
        """Return the F matrix"""
        return self._F

    @property
    def G(self):
        """Return the G matrix"""
        return self._G

    @property
    def W(self):
        """Return the W matrix"""
        return self._W

    def __add__(self, other):
        # type: (UnivariateStructure) -> UnivariateStructure
        F = np.vstack((self.F, other.F))
        G = block_diag(*[self.G, other.G])
        W = block_diag(*[self.W, other.W])

        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def locally_constant(W: float) -> UnivariateStructure:
        """Create a locally constant univariate structure"""
        return UnivariateStructure(
            F=np.matrix([[1]]), G=np.matrix([[1]]), W=np.matrix([[W]])
        )

    @staticmethod
    def locally_linear(W: np.matrix) -> UnivariateStructure:
        """Create a locally linear univariate structure"""
        F = np.matrix([[1], [0]])
        G = np.matrix([[1, 1], [0, 1]])
        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def arma(p: int, betas: np.ndarray, W: float) -> UnivariateStructure:
        """Create a ARMA(p) structure"""
        if p < 1:
            raise ValueError("`p` must be 1 or higher.")
        if len(betas) != p:
            raise ValueError("`betas` must have `p` elements.")

        F = np.array([[1.0] + [0.0] * (p - 1)]).transpose()
        if p == 1:  # can simplify for ARMA(1)
            G = np.identity(p) * betas[0]
        else:
            G = np.identity(p - 1)
            G = np.hstack((G, np.zeros((p - 1, 1))))
            G = np.vstack((betas, G))
        _W = np.zeros((p, p))
        _W[0, 0] = W
        return UnivariateStructure(F=F, G=G, W=_W)

    @staticmethod
    def cyclic_fourier(
        period: int, harmonics: int, W: np.matrix
    ) -> UnivariateStructure:
        """Create a seasonal (Fourier) structure"""
        omega = 2.0 * np.pi / period
        harmonic1: np.ndarray = np.identity(2) * cos(omega)
        harmonic1[0, 1] = sin(omega)
        harmonic1[1, 0] = -harmonic1[0, 1]

        _G: List[np.ndarray] = [np.empty([1, 1])] * harmonics
        _G[0] = np.copy(harmonic1)
        if harmonics > 1:
            for i in np.arange(1, harmonics):
                _G[i] = np.dot(_G[i - 1], harmonic1)
            G = block_diag(*_G)
        else:
            G = harmonic1

        F = np.array([[1.0, 0.0] * harmonics]).transpose()

        return UnivariateStructure(F=F, G=G, W=W)


class MultivariateStructure:
    """Create a multivariate structure"""

    def __init__(self, F, G, W):
        self._F = F
        self._G = G
        self._W = W

    @property
    def F(self):
        """Return the F matrix"""
        return self._F

    @property
    def G(self):
        """Return the G matrix"""
        return self._G

    @property
    def W(self):
        """Return the W matrix"""
        return self._W

    @staticmethod
    def build(*dglms):
        """Build the structure"""
        _F = block_diag(*[model.F for model in dglms])
        _G = block_diag(*[model.G for model in dglms])
        _W = block_diag(*[model.W for model in dglms])
        return MultivariateStructure(F=_F, G=_G, W=_W)
