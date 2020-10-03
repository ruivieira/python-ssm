"""Module defining different types of Dynamic Generalised Linear Models"""
from abc import ABC, abstractmethod
import math
import numpy as np
import numpy.random as rand

# from numpy.random import multivariate_normal as mvn
# from numpy.random import normal
# from numpy.random import poisson

from pssm.utils import ilogit
from pssm.structure import MultivariateStructure


class DLM(ABC):
    """Abstract Dynamic Linear Model class"""

    def __init__(self, structure, state_prior=None):
        self._structure = structure
        self._Ft = np.transpose(structure.F)
        if state_prior is not None:
            self._current_state = state_prior
        else:
            self._current_state = np.zeros(structure.W.shape[0])

    @property
    def structure(self):
        """Returns the structure of this DLM"""
        return self._structure

    @property
    def current_state(self):
        """Returns the current state of this DLM"""
        return self._current_state

    @abstractmethod
    def _eta(self, _lambda):
        pass

    @abstractmethod
    def _sample_obs(self, mean):
        pass

    def state(self, previous):
        """Calculate new state based on the previous"""
        mean = np.atleast_1d(
            np.squeeze(np.asarray(np.dot(self._structure.G, previous)))
        )
        return rand.multivariate_normal(mean=mean, cov=self._structure.W)

    def observation(self, state):
        """Generate an observation given the state"""
        mean = self._eta(np.dot(self._Ft, state))
        return self._sample_obs(mean)

    def next(self):
        """Iterator method"""
        return self.__next__()

    def __next__(self):
        """Iterator to generate the next state"""
        self._current_state = self.state(self._current_state)
        obs = self.observation(self._current_state)
        return self._current_state, obs


class NormalDLM(DLM):
    """
    An instance of a Normal DLM
    """

    def __init__(self, structure, V, state_prior=None):
        super().__init__(structure=structure, state_prior=state_prior)
        self._V = np.matrix([[V]])

    def _eta(self, _lambda):
        return _lambda

    def _sample_obs(self, mean):
        y = rand.normal(loc=mean, scale=self._V)
        return y if np.isscalar(y) else np.asscalar(y)


class PoissonDLM(DLM):
    """
    An instance of a Poisson DLM
    """

    def __init__(self, structure, state_prior=None):
        super().__init__(structure=structure, state_prior=state_prior)

    def _eta(self, _lambda):
        return math.exp(_lambda)

    def _sample_obs(self, mean):
        return rand.poisson(mean)


class BinomialDLM(DLM):
    """
    An instance of a Binomial DLM
    """

    def __init__(self, structure, categories=1, state_prior=None):
        super().__init__(structure=structure, state_prior=state_prior)
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
        super().__init__(
            MultivariateStructure.build(*[dglm.structure for dglm in dglms])
        )
        self._dglms = dglms

    def observation(self, state):
        lambdas = np.dot(self._Ft, state)
        ys = []
        for i in range(len(self._dglms)):
            dglm = self._dglms[i]
            eta = dglm._eta(lambdas[i])  # pylint: disable=protected-access
            ys.append(dglm._sample_obs(eta))  # pylint: disable=protected-access

        return np.array(ys)
