import numpy as np


class DGLM:
    pass

class NormalDLM(DGLM):
    def __init__(self, structure, V):
        self._structure = structure
        self._V = np.matrix([[V]])

    def state(self, previous):
        mean = self._structure.G.dot(previous).A1
        return np.random.multivariate_normal(mean, self._structure.W)

    def observation(self, state):
        mean = self._structure.F.dot(state).A1
        return np.random.multivariate_normal(mean, self._V)