# coding: utf-8


import unittest
from onlineldavb import onlineldavb
from scipy.special import gammaln, psi
import numpy as np
import numpy.testing as npt


class TestOnlineLDA(unittest.TestCase):
    def test_dirichlet_expectation(self):
        """
        Check if expectation is calculated correctly.
        this is basically to guide refactoring
        """
        rdata = np.random.randint(1,10,20)
        alpha = rdata/sum(rdata)#Normalizing
        if len(alpha.shape) == 1:
            e = psi(alpha) - psi(np.sum(alpha))
        else:
            e = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]

        npt.assert_array_equal(e, onlineldavb.dirichlet_expectation(alpha))
