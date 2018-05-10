"""composite_tests

This module contains tests for the Composite DGLM models
"""
import unittest

from nose.tools import assert_equals, assert_almost_equal
from nose.tools.nontrivial import raises

from pssm.dglm import NormalDLM, CompositeDLM
from pssm.structure import UnivariateStructure
import numpy as np


class CompositeTests(unittest.TestCase):
    """Test suite for composite DGLM structures
    """

    def composite_lc_structure_dimension_test(self):
        """Test if the composite LC structure has the expected dimensions.
        """
        lc1 = UnivariateStructure.locally_constant(1.0)
        lc2 = UnivariateStructure.locally_constant(1.0)

        ndlm1 = NormalDLM(lc1, 1.0)
        ndlm2 = NormalDLM(lc2, 1.0)

        composite = CompositeDLM(ndlm1, ndlm2)

        shape = composite.structure.F.shape
        assert_equals(shape, (2,2), "F dimensions not correct")
        shape = composite.structure.G.shape
        assert_equals(shape, (2,2), "G dimensions not correct")
        shape = composite.structure.W.shape
        assert_equals(shape, (2,2), "W dimensions not correct")

    def composite_lc_state_dimension_test(self):
        """Test if the composite LC state has the expected dimensions.
        """
        lc1 = UnivariateStructure.locally_constant(1.0)
        lc2 = UnivariateStructure.locally_constant(1.0)

        ndlm1 = NormalDLM(lc1, 1.0)
        ndlm2 = NormalDLM(lc2, 1.0)

        composite = CompositeDLM(ndlm1, ndlm2)

        m0 = np.array([0, 0])
        state = composite.state(m0)
        shape = state.shape
        assert_equals(shape, (2,), "state dimensions not correct")

    def composite_lc_obs_dimension_test(self):
        """Test if the composite LC observation has the expected dimensions.
        """
        lc1 = UnivariateStructure.locally_constant(1.0)
        lc2 = UnivariateStructure.locally_constant(1.0)

        ndlm1 = NormalDLM(lc1, 1.0)
        ndlm2 = NormalDLM(lc2, 1.0)

        composite = CompositeDLM(ndlm1, ndlm2)

        m0 = np.array([0, 0])
        state = composite.state(m0)
        obs = composite.observation(state)
        shape = obs.shape
        assert_equals(shape, (2,), "observation dimensions not correct")

    def composite_lcll_structure_dimension_test(self):
        """Test if the composite LC/LL structure has the expected dimensions.
        """
        lc = UnivariateStructure.locally_constant(1.0)
        ll = UnivariateStructure.locally_linear(np.eye(2))

        ndlm1 = NormalDLM(lc, 1.0)
        ndlm2 = NormalDLM(ll, 1.0)

        composite = CompositeDLM(ndlm1, ndlm2)

        shape = composite.structure.F.shape
        assert_equals(shape, (3,2), "F dimensions not correct")
        shape = composite.structure.G.shape
        assert_equals(shape, (3,3), "G dimensions not correct")
        shape = composite.structure.W.shape
        assert_equals(shape, (3,3), "W dimensions not correct")

    def composite_lcll_state_dimension_test(self):
        """Test if the composite LC/LL state has the expected dimensions.
        """
        lc = UnivariateStructure.locally_constant(1.0)
        ll = UnivariateStructure.locally_linear(np.eye(2))

        ndlm1 = NormalDLM(lc, 1.0)
        ndlm2 = NormalDLM(ll, 1.0)

        composite = CompositeDLM(ndlm1, ndlm2)

        m0 = np.array([0, 0, 0])
        state = composite.state(m0)
        assert_equals(state.shape, (3,), "state dimensions not correct")

    def composite_lcll_obs_dimension_test(self):
        """Test if the composite LC/LL observation has the expected dimensions.
        """
        lc = UnivariateStructure.locally_constant(1.0)
        ll = UnivariateStructure.locally_linear(np.eye(2))

        ndlm1 = NormalDLM(lc, 1.0)
        ndlm2 = NormalDLM(ll, 1.0)

        composite = CompositeDLM(ndlm1, ndlm2)

        m0 = np.array([0, 0, 0])
        state = composite.state(m0)
        obs = composite.observation(state)
        assert_equals(obs.shape, (2,), "observation dimensions not correct")