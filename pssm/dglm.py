import abc
import numpy as np
from numpy.random import multivariate_normal as mvn
from numpy.random import normal
from numpy.random import poisson
import math

from pssm.utils import ilogit
from pssm.structure import MultivariateStructure

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class DLM(ABC):

    def __init__(self, structure, state_prior=None):
        self._structure = structure
        self._Ft = np.transpose(structure.F)
        if state_prior is not None:
            self._current_state = state_prior
        else:
            self._current_state = np.zeros(structure.W.shape[0])

    @property
    def structure(self):
        return self._structure

    @property
    def current_state(self):
        return self._current_state

    @abc.abstractmethod
    def _eta(self, _lambda):
        pass

    @abc.abstractmethod
    def _sample_obs(self, mean):
        pass

    def state(self, previous):
        mean = np.atleast_1d(np.squeeze(np.asarray(
            np.dot(self._structure.G, previous))))
        return mvn(mean=mean, cov=self._structure.W)

    def observation(self, state):
        mean = self._eta(np.dot(self._Ft, state))
        return self._sample_obs(mean)

    def next(self):
        return self.__next__()

    def __next__(self):
        self._current_state = self.state(self._current_state)
        obs = self.observation(self._current_state)
        return self._current_state, obs


class NormalDLM(DLM):
    """
    An instance of a Normal DLM
    """

    def __init__(self, structure, V, state_prior=None):
        super(NormalDLM, self).__init__(structure=structure,
                                        state_prior=state_prior)
        self._V = np.matrix([[V]])

    def _eta(self, _lambda):
        return _lambda

    def _sample_obs(self, mean):
        return np.asscalar(normal(loc=mean, scale=self._V))


class PoissonDLM(DLM):
    """
    An instance of a Poisson DLM
    """

    def __init__(self, structure, state_prior=None):
        super(PoissonDLM, self).__init__(structure=structure,
                                         state_prior=state_prior)

    def _eta(self, _lambda):
        return math.exp(_lambda)

    def _sample_obs(self, mean):
        return poisson(mean)


class BinomialDLM(DLM):
    """
    An instance of a Binomial DLM
    """

    def __init__(self, structure, categories=1, state_prior=None):
        super(BinomialDLM, self).__init__(structure=structure,
                                          state_prior=state_prior)
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
        super(CompositeDLM, self).__init__(
            MultivariateStructure.build(*[dglm.structure for dglm in dglms]))
        self._dglms = dglms

    def observation(self, state):
        lambdas = np.dot(self._Ft, state)
        ys = []
        for i in range(len(self._dglms)):
            dglm = self._dglms[i]
            eta = dglm._eta(lambdas[i])
            ys.append(dglm._sample_obs(eta))

        return np.array(ys)
