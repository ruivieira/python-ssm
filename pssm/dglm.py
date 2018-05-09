import abc
import numpy as np
from scipy.stats import multivariate_normal as mvn, poisson
import math

from pssm.utils import ilogit

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class DLM(ABC):

    def __init__(self, structure):
        self._structure = structure
        self._Ft = np.transpose(structure.F)

    def state(self, previous):
        mean = np.squeeze(np.asarray(np.dot(self._structure.G, previous)))
        return mvn(mean=mean, cov=self._structure.W, allow_singular=True).rvs()

    @abc.abstractmethod
    def observation(self, state): pass


class NormalDLM(DLM):
    """
    An instance of a Normal DLM
    """

    def __init__(self, structure, V):
        super(NormalDLM, self).__init__(structure)
        self._V = np.matrix([[V]])

    def observation(self, state):
        mean = np.dot(self._Ft, state)
        return mvn(mean=mean, cov=self._V).rvs()


class PoissonDLM(DLM):
    """
    An instance of a Poisson DLM
    """

    def __init__(self, structure):
        super(PoissonDLM, self).__init__(structure)

    def observation(self, state):
        mean = math.exp(np.asscalar(np.dot(self._Ft, state)))
        return poisson(mean).rvs()


class BinomialDLM(DLM):
    """
    An instance of a Binomial DLM
    """

    def __init__(self, structure, categories=1):
        super(BinomialDLM, self).__init__(structure)
        self._categories = categories

    def observation(self, state):
        eta = ilogit(np.asscalar(np.dot(self._Ft, state)))
        return np.random.binomial(n=self._categories, p=eta)
