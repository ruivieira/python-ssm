"""structure_tests

This module contains tests for the DGLM structures
"""
# pylint: disable=no-self-use
import unittest
from nose.tools import assert_equals, assert_almost_equal  # type: ignore
from nose.tools.nontrivial import raises  # type: ignore
import numpy as np
from pssm.dglm import NormalDLM
from pssm.structure import UnivariateStructure


class StructureTests(unittest.TestCase):
    """Test suite for DGLM structures"""

    def univariate_lc_structure_dimension_test(self):
        """Test if the LC structure has the expected dimensions."""
        lc = UnivariateStructure.locally_constant(1.0)
        shape = lc.F.shape
        assert_equals(shape[0], 1, "F dimensions not correct")
        assert_equals(shape[1], 1, "F dimensions not correct")
        shape = lc.G.shape
        assert_equals(shape[0], 1, "G dimensions not correct")
        assert_equals(shape[1], 1, "G dimensions not correct")
        shape = lc.W.shape
        assert_equals(shape[0], 1, "W dimensions not correct")
        assert_equals(shape[1], 1, "W dimensions not correct")

    def univariate_lc_structure_values_test(self):
        """Test if the LC structure has the expected values"""
        lc = UnivariateStructure.locally_constant(1.3)
        assert_equals(lc.F[0, 0], 1, "F value not correct")
        assert_equals(lc.G[0, 0], 1, "G value not correct")
        assert_equals(lc.W[0, 0], 1.3, "W value not correct")

    @raises(ValueError)
    def univariate_lc_state_dimension_mismatch_test(self):
        """
        Test if a state of wrong dimensions throws an exception
        """
        lc = UnivariateStructure.locally_constant(1.0)
        dlm = NormalDLM(structure=lc, V=1.4)
        state0 = np.array([0, 0])
        dlm.observation(state0)

    def univariate_ll_structure_values_test(self):
        """Test if the LL structure has the expected values"""
        ll = UnivariateStructure.locally_linear(np.matrix([[1.3, 0], [0, 0.4]]))
        assert_equals(ll.F[0, 0], 1, "F value not correct")
        assert_equals(ll.F[1, 0], 0, "F value not correct")
        assert_equals(ll.G[0, 0], 1, "G value not correct")
        assert_equals(ll.G[0, 1], 1, "G value not correct")
        assert_equals(ll.G[1, 0], 0, "G value not correct")
        assert_equals(ll.G[1, 1], 1, "G value not correct")
        assert_equals(ll.W[0, 0], 1.3, "W value not correct")
        assert_equals(ll.W[0, 1], 0, "W value not correct")
        assert_equals(ll.W[1, 0], 0, "W value not correct")
        assert_equals(ll.W[1, 1], 0.4, "W value not correct")

    def univariate_ll_structure_dimension_test(self):
        """Test if the LL structure has the expected dimensions."""
        ll = UnivariateStructure.locally_linear(np.matrix([[1.3, 0], [0, 0.4]]))

        shape = ll.F.shape
        assert_equals(shape[0], 2, "F dimensions not correct")
        assert_equals(shape[1], 1, "F dimensions not correct")
        shape = ll.G.shape
        assert_equals(shape[0], 2, "G dimensions not correct")
        assert_equals(shape[1], 2, "G dimensions not correct")
        shape = ll.W.shape
        assert_equals(shape[0], 2, "W dimensions not correct")
        assert_equals(shape[1], 2, "W dimensions not correct")

    def univariate_arma1_structure_dimension_test(self):
        """Test if the ARMA(1) structure has the expected dimensions."""
        arma1 = UnivariateStructure.arma(p=1, betas=[0.2], W=0.3)
        assert_equals(arma1.F.shape, (1, 1), "F dimensions not correct")
        assert_equals(arma1.G.shape, (1, 1), "G dimensions not correct")
        assert_equals(arma1.W.shape, (1, 1), "W dimensions not correct")

    def univariate_arma2_structure_dimension_test(self):
        """Test if the ARMA(2) structure has the expected dimensions."""
        arma1 = UnivariateStructure.arma(p=2, betas=[0.2] * 2, W=0.3)
        assert_equals(arma1.F.shape, (2, 1), "F dimensions not correct")
        assert_equals(arma1.G.shape, (2, 2), "G dimensions not correct")
        assert_equals(arma1.W.shape, (2, 2), "W dimensions not correct")

    def univariate_arma10_structure_dimension_test(self):
        """Test if the ARMA(10) structure has the expected dimensions."""
        arma1 = UnivariateStructure.arma(p=10, betas=[0.2] * 10, W=0.3)
        assert_equals(arma1.F.shape, (10, 1), "F dimensions not correct")
        assert_equals(arma1.G.shape, (10, 10), "G dimensions not correct")
        assert_equals(arma1.W.shape, (10, 10), "W dimensions not correct")

    @raises(ValueError)
    def univariate_arma_invalid_p_test(self):
        """Invalid p in ARMA(p) must raise ValueError."""
        UnivariateStructure.arma(p=0, betas=[0.2], W=0.3)

    @raises(ValueError)
    def univariate_arma_betas_mismatch_test(self):
        """Betas dimension mismatch must raise ValueError"""
        UnivariateStructure.arma(p=2, betas=[0.2] * 3, W=0.3)

    def ll_lc_composition_dimension_test(self):
        """Test the dimensions of composing a LL and LC structure"""

        lc = UnivariateStructure.locally_constant(1.0)
        ll = UnivariateStructure.locally_linear(np.matrix([[1.3, 0], [0, 0.4]]))

        composed = lc + ll
        assert_equals(composed.F.shape, (3, 1), "F dimensions not correct")
        assert_equals(composed.G.shape, (3, 3), "G dimensions not correct")
        assert_equals(composed.W.shape, (3, 3), "W dimensions not correct")

    def fourier_values_test(self):
        """Test if the fourier structure has the expected values"""
        c = 0.80901699
        s = 0.58778525
        W = np.identity(2)
        structure = UnivariateStructure.cyclic_fourier(10, 1, W)
        assert_equals(structure.F[0, 0], 1, "F value not correct")
        assert_equals(structure.F[1, 0], 0, "F value not correct")
        assert_almost_equal(structure.G[0, 0], c, places=7)
        assert_almost_equal(structure.G[0, 1], s, places=7)
        assert_almost_equal(structure.G[1, 0], -s, places=7)
        assert_almost_equal(structure.G[1, 1], c, places=7)

    def composition_test(self):
        """Test if a complex composition has the correct dimensions"""
        s1 = UnivariateStructure.locally_constant(0.01)
        s2 = UnivariateStructure.cyclic_fourier(30, 5, np.eye(10) * 0.1)
        s3 = UnivariateStructure.cyclic_fourier(365, 5, np.eye(10) * 1.7)
        s = s1 + s2 + s3
        assert_equals(s.F.shape, (21, 1), "F dimensions not correct")
        assert_equals(s.G.shape, (21, 21), "G dimensions not correct")
        assert_equals(s.W.shape, (21, 21), "W dimensions not correct")
