from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal as mvn


class DLM:

    __metaclass__ = ABCMeta

    def __init__(self, structure):
        self._structure = structure
        self._Ft = np.transpose(structure.F)

    def state(self, previous):
        mean = np.squeeze(np.asarray(np.dot(self._structure.G, previous)))
        return mvn(mean=mean, cov=self._structure.W).rvs()

    @abstractmethod
    def observation(self, state): pass


class NormalDLM(DLM):
    """
    An instance of a Normal DLM
    """
    def __init__(self, structure, V):
        super().__init__(structure)
        self._V = np.matrix([[V]])

    def state(self, previous):
        mean = np.squeeze(np.asarray(np.dot(self._structure.G, previous)))
        return mvn(mean=mean, cov=self._structure.W).rvs()

    def observation(self, state):
        mean = np.dot(self._Ft, state)
        return mvn(mean=mean, cov=self._V).rvs()
