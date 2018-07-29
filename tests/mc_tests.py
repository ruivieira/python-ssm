"""structure_tests

This module contains tests for the DGLM structures
"""
import unittest

from nose.tools import assert_equals

import numpy as np
from numpy.random import multivariate_normal as mvn

from pssm.mc import FFBS
from pssm.filters import KalmanFilter
from pssm.dglm import NormalDLM
from pssm.structure import UnivariateStructure


class FFBSTests(unittest.TestCase):
    """Test suite for FFBS
    """

    @classmethod
    def setUpClass(cls):
        # Locally constant
        lc = {'structure': UnivariateStructure.locally_constant(1.0)}
        lc['model'] = NormalDLM(structure=lc['structure'], V=1.4)
        lc['nobs'] = 100
        # the initial state prior
        m0 = np.array([0])
        C0 = np.matrix([[1]])
        state0 = mvn(m0, C0)

        lc['states'] = [state0]

        for t in range(1, lc['nobs']):
            lc['states'].append(lc['model'].state(lc['states'][t - 1]))

        lc['obs'] = [None]
        for t in range(1, lc['nobs']):
            lc['obs'].append(lc['model'].observation(lc['states'][t]))

        cls._lc = lc

        # Locally linear
        ll = {'structure': UnivariateStructure.locally_linear(np.identity(2))}
        ll['model'] = NormalDLM(structure=ll['structure'], V=1.4)
        ll['nobs'] = 100
        # the initial state prior
        m0 = np.array([0, 0])
        C0 = np.identity(2)
        state0 = mvn(m0, C0)

        ll['states'] = [state0]

        for t in range(1, ll['nobs']):
            ll['states'].append(ll['model'].state(ll['states'][t - 1]))

        ll['obs'] = [None]
        for t in range(1, ll['nobs']):
            ll['obs'].append(ll['model'].observation(ll['states'][t]))

        cls._ll = ll

    def univariate_lc_bs_elements_test(self):
        """Test how many states LC BS returns
        """

        # Run Kalman filter
        ms = [np.array([0])]
        Cs = [np.identity(1) * 1000]

        kf = KalmanFilter(structure=FFBSTests._lc['structure'], V=1.4)
        for t in range(1, FFBSTests._lc['nobs']):
            m, C = kf.filter(FFBSTests._lc['obs'][t], ms[t - 1], Cs[t - 1])
            ms.append(m)
            Cs.append(C)

        states = FFBS.backward_sampling(ms, Cs, np.identity(1))
        assert_equals(len(states), FFBSTests._lc['nobs'] - 1)

    def univariate_lc_bs_dimension_test(self):
        """Test states LC BS dimension
        """

        # Run Kalman filter
        ms = [np.array([0])]
        Cs = [np.identity(1) * 1000]

        kf = KalmanFilter(structure=FFBSTests._lc['structure'], V=1.4)
        for t in range(1, FFBSTests._lc['nobs']):
            m, C = kf.filter(FFBSTests._lc['obs'][t], ms[t - 1], Cs[t - 1])
            ms.append(m)
            Cs.append(C)

        states = FFBS.backward_sampling(ms, Cs, np.identity(1))
        assert_equals(states[50].shape, (1,))

    def univariate_ll_bs_elements_test(self):
        """Test how many states LL BS returns
        """

        # Run Kalman filter
        ms = [np.array([0, 0])]
        Cs = [np.identity(2) * 1000]

        kf = KalmanFilter(structure=FFBSTests._ll['structure'], V=1.4)
        for t in range(1, FFBSTests._ll['nobs']):
            m, C = kf.filter(FFBSTests._ll['obs'][t], ms[t - 1], Cs[t - 1])
            ms.append(m)
            Cs.append(C)

        states = FFBS.backward_sampling(ms, Cs, np.identity(2))
        assert_equals(len(states), FFBSTests._ll['nobs'] - 1)

    def univariate_ll_bs_dimension_test(self):
        """Test states LL BS dimensions
        """

        # Run Kalman filter
        ms = [np.array([0, 0])]
        Cs = [np.identity(2) * 1000]

        kf = KalmanFilter(structure=FFBSTests._ll['structure'], V=1.4)
        for t in range(1, FFBSTests._ll['nobs']):
            m, C = kf.filter(FFBSTests._ll['obs'][t], ms[t - 1], Cs[t - 1])
            ms.append(m)
            Cs.append(C)

        states = FFBS.backward_sampling(ms, Cs, np.identity(2))
        assert_equals(states[0].shape, (2,))
