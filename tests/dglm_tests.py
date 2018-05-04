"""dglm_tests

This module contains tests for the DGLMs
"""
import unittest

from nose.tools import assert_true

from pssm.dglm import NormalDLM, PoissonDLM
from pssm.structure import UnivariateStructure
import numpy as np


class DglmTests(unittest.TestCase):
    """Test suite for DGLM observations
    """

    def gaussian_univariate_lc_y_dimension_test(self):
        """Test if the Normal LC DLM observation has the expected dimension.
        """
        lc = UnivariateStructure.locally_constant(1.0)
        dlm = NormalDLM(structure=lc, V=1.4)
        y = dlm.observation(np.array([0]))
        assert_true(np.isscalar(y), "Observation is not a scalar")

    def poisson_univariate_lc_y_dimension_test(self):
        """Test if the Poisson LC DLM observation has the expected dimension.
        """
        lc = UnivariateStructure.locally_constant(1.0)
        dglm = PoissonDLM(structure=lc)
        state0 = np.array([0])
        y = dglm.observation(state0)
        assert_true(np.isscalar(y), "Observation is not a scalar")
