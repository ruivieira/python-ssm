"""filter_tests

This module contains tests for the filtering methods
"""
# pylint: disable=no-self-use
import unittest

from nose.tools import assert_equals  # type: ignore

import numpy as np
from pssm.filters import KalmanFilter

from pssm.structure import UnivariateStructure


class StructureTests(unittest.TestCase):
    """Test suite for DGLM structures"""

    def univariate_lc_filter_dimension_test(self):
        """Test if the KF output has LC dimensions."""
        lc = UnivariateStructure.locally_constant(1.0)

        m0 = np.array([0])
        C0 = np.diag([1e3])

        m, C = KalmanFilter(structure=lc, V=2.5).filter(y=0.0, m=m0, C=C0)

        assert_equals(m.shape, (1,), "First moment dimensions are wrong")
        assert_equals(C.shape, (1, 1), "Second moment dimensions are wrong")

    def univariate_ll_filter_dimension_test(self):
        """Test if the KF output has LL dimensions."""
        ll = UnivariateStructure.locally_linear(W=np.matrix([[0.1, 0], [0, 0.1]]))
        print(ll.F.shape)

        m0 = np.array([0, 0])
        C0 = np.diag([1e3, 1e3])
        m, C = KalmanFilter(structure=ll, V=2.5).filter(y=0.0, m=m0, C=C0)

        assert_equals(m.shape, (2,), "First moment dimensions are wrong")
        assert_equals(C.shape, (2, 2), "Second moment dimensions are wrong")
