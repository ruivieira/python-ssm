import numpy as np
from scipy.stats import multivariate_normal as mvn


class DGLM:
    pass


class NormalDLM(DGLM):
    """
    An instance of a Normal DLM
    """
    def __init__(self, structure, V):
        self._structure = structure
        self._V = np.matrix([[V]])
        self._Ft = np.transpose(structure.F)

    def state(self, previous):
        mean = np.squeeze(np.asarray(np.dot(self._structure.G, previous)))
        return mvn(mean=mean, cov=self._structure.W).rvs()

    def observation(self, state):
        mean = np.dot(self._Ft, state)
        return mvn(mean=mean, cov=self._V).rvs()
