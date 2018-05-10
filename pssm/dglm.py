import abc
import numpy as np
from scipy.stats import multivariate_normal as mvn, poisson
import math

from pssm.utils import ilogit
from pssm.structure import MultivariateStructure

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class DLM(ABC):

    def __init__(self, structure):
        self._structure = structure
        self._Ft = np.transpose(structure.F)

    @property
    def structure(self):
        return self._structure


    @abc.abstractmethod
    def _eta(self, _lambda): pass

    @abc.abstractmethod
    def _sample_obs(self, mean): pass

    def state(self, previous):
        mean = np.squeeze(np.asarray(np.dot(self._structure.G, previous)))
        return mvn(mean=mean, cov=self._structure.W, allow_singular=True).rvs()

    def observation(self, state):
        mean = self._eta(np.dot(self._Ft, state))
        return self._sample_obs(mean)


class NormalDLM(DLM):
    """
    An instance of a Normal DLM
    """

    def __init__(self, structure, V):
        super(NormalDLM, self).__init__(structure)
        self._V = np.matrix([[V]])

    def _eta(self, _lambda):
        return _lambda

    def _sample_obs(self, mean):
        return mvn(mean=mean, cov=self._V).rvs()


class PoissonDLM(DLM):
    """
    An instance of a Poisson DLM
    """

    def __init__(self, structure):
        super(PoissonDLM, self).__init__(structure)

    def _eta(self, _lambda):
        return math.exp(_lambda)

    def _sample_obs(self, mean):
        return poisson(mean).rvs()


class BinomialDLM(DLM):
    """
    An instance of a Binomial DLM
    """

    def __init__(self, structure, categories=1):
        super(BinomialDLM, self).__init__(structure)
        self._categories = categories

    def _eta(self, _lambda):
        return ilogit(_lambda)

    def _sample_obs(self, mean):
        return np.random.binomial(n=self._categories, p=mean)


class CompositeDLM(DLM):
    """
    A multivariate composite DLM
    """

    def _sample_obs(self, mean):
        pass

    def _eta(self, _lambda):
        pass

    def __init__(self, *dglms):
        super(CompositeDLM, self).__init__(MultivariateStructure.build(*[dglm.structure for dglm in dglms]))
        self._dglms = dglms

    def observation(self, state):
        lambdas = np.dot(self._Ft, state)
        ys = []
        for i in range(len(self._dglms)):
            dglm = self._dglms[i]
            eta = dglm._eta(lambdas[i])
            ys.append(dglm._sample_obs(eta))

        return np.array(ys)