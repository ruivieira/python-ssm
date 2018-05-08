import numpy as np
from scipy.linalg import block_diag


class UnivariateStructure:
    def __init__(self, F, G, W):
        self._F = F
        self._G = G
        self._W = W

    @property
    def F(self):
        return self._F

    @property
    def G(self):
        return self._G

    @property
    def W(self):
        return self._W

    def __add__(self, other):
        # type: (UnivariateStructure) -> UnivariateStructure
        F = np.vstack((self.F, other.F))
        G = block_diag(*[self.G, other.G])
        W = block_diag(*[self.W, other.W])

        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def locally_constant(W):
        # type: (float) -> UnivariateStructure
        F = np.matrix([[1]])
        G = np.matrix([[1]])
        W = np.matrix([[W]])
        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def locally_linear(W):
        # type: (ndarray) -> UnivariateStructure
        F = np.matrix([[1], [0]])
        G = np.matrix([[1, 1], [0, 1]])
        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def arma(p, betas, W):
        # type: (ndarray, List[float], float) -> UnivariateStructure
        if p < 1:
            raise ValueError("`p` must be 1 or higher.")
        if len(betas) != p:
            raise ValueError("`betas` must have `p` elements.")

        F = np.transpose(np.matrix([[1.0] + [0.0] * (p - 1)]))
        if p == 1:  # can simplify for ARMA(1)
            G = np.identity(p) * betas[0]
        else:
            G = np.identity(p-1)
            G = np.hstack((G, np.zeros((p-1, 1))))
            G = np.vstack((betas, G))
        _W = np.zeros((p, p))
        _W[0, 0] = W
        return UnivariateStructure(F=F, G=G, W=_W)

    @staticmethod
    def cyclic_fourier(period, harmonics, W):
        # type: (int, int, ndarray) -> UnivariateStructure
        om = 2.0 * np.pi / period
        harmonic1 = np.identity(2) * np.cos(om)
        harmonic1[0, 1] = np.sin(om)
        harmonic1[1, 0] = -harmonic1[0, 1]

        _G = [None] * harmonics
        _G[0] = np.copy(harmonic1)
        if harmonics > 1:
            for i in np.arange(1, harmonics):
                _G[i] = np.dot(_G[i - 1], harmonic1)
            G = block_diag(*_G)
        else:
            G = harmonic1

        F = np.transpose(np.matrix([[1.0, 0.0] * harmonics]))

        return UnivariateStructure(F=F, G=G, W=W)
