import numpy as np
from scipy.linalg import block_diag


class UnivariateStructure:
    def __init__(self, F, G, W):
        self.F = F
        self.G = G
        self.W = W

    def __add__(self, other):
        F = np.hstack((self.F, other.F))
        G = block_diag(*[self.G, other.G])
        W = block_diag(*[self.W, other.W])

        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def locally_constant(W):
        F = np.matrix([[1]])
        G = np.matrix([[1]])
        W = np.matrix([[W]])
        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def locally_linear(W):
        F = np.matrix([[1, 0]])
        G = np.matrix([[1, 1], [0, 1]])
        return UnivariateStructure(F=F, G=G, W=W)

    @staticmethod
    def cyclic_fourier(period, harmonics, W):
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

        F = np.matrix([[1.0, 0.0] * harmonics])

        return UnivariateStructure(F=F, G=G, W=W)