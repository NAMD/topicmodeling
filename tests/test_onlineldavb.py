# coding: utf-8


import unittest
from onlineldavb import onlineldavb
from scipy.special import gammaln, psi
import numpy as np
import numpy.testing as npt
import random


class TestOnlineLDA(unittest.TestCase):
    def setUp(self):
        with open('data/sampledoc') as f:
            self.sampledoc = f.read()

    def test_dirichlet_expectation_unidimensional(self):
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

    def test_dirichlet_expectation_bidimensional(self):
        """
        Check if expectation is calculated correctly.
        this is basically to guide refactoring
        """
        rdata = np.random.randint(1,10,(20,20))
        alpha = rdata/sum(rdata)#Normalizing
        if len(alpha.shape) == 1:
            e = psi(alpha) - psi(np.sum(alpha))
        else:
            e = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]

        npt.assert_array_equal(e, onlineldavb.dirichlet_expectation(alpha))

    def test_parse_docs_returning_the_right_types_single_doc(self):
        vocab = {w: n for n, w in enumerate(self.sampledoc.split())}
        docs = [' '.join(random.sample(vocab.keys(), 20)) for i in range(1)]
        ids, cts = onlineldavb.parse_doc_list(docs, vocab)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(cts, list)
        self.assertIsInstance(ids[0][0], int)
        self.assertIsInstance(cts[0][0], int)

    def test_parse_docs_returning_the_right_types_multiple_docs(self):
        vocab = {w: n for n, w in enumerate(self.sampledoc.split())}
        docs = [' '.join(random.sample(vocab.keys(), 20)) for i in range(15)]
        ids, cts = onlineldavb.parse_doc_list(docs, vocab)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(cts, list)
        self.assertIsInstance(ids[0][0], int)
        self.assertIsInstance(cts[0][0], int)

    def test_olda_update_lambda(self):
        vocab = {w: n for n, w in enumerate(self.sampledoc.split())}
        docs = [' '.join(random.sample(vocab.keys(), 20)) for i in xrange(64)]
        K = 5
        D = 64
        olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
        gamma, bound = olda.update_lambda(docs)
        self. assertIsInstance(gamma, float)
        self. assertIsInstance(bound, float)

