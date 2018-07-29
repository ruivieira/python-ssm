"""dglm_tests

This module contains tests for the DGLMs
"""
import unittest

from nose.tools import assert_true, assert_equal
import numpy as np
from pssm.dglm import NormalDLM, PoissonDLM
from pssm.structure import UnivariateStructure


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

    def univariate_lc_prior_dimensions_test(self):
        """The default LC DLM state prior should have the correct dimensions
        """
        lc = UnivariateStructure.locally_constant(1.0)
        dlm = NormalDLM(structure=lc, V=1.4)
        assert_equal(dlm.current_state.shape, (1,),
                     "Prior has the wrong dimensions")

    def univariate_ll_prior_dimensions_test(self):
        """The default LL DLM state prior should have the correct dimensions
        """
        ll = UnivariateStructure.locally_linear(np.eye(2))
        dlm = NormalDLM(structure=ll, V=1.4)
        assert_equal(dlm.current_state.shape, (2,),
                     "Prior has the wrong dimensions")

    def univariate_ll_iterator_dimensions_test(self):
        """The LL DLM iterator result should have the correct dimensions
        """
        ll = UnivariateStructure.locally_linear(np.eye(2))
        dlm = NormalDLM(structure=ll, V=1.4)
        items = [next(dlm) for _ in range(10)]
        print(items)
        assert_equal(items[0][0].shape, (2,),
                     "State has the wrong dimensions")
        assert_true(np.isscalar(items[0][1]),
                    "Observation has the wrong dimensions")
