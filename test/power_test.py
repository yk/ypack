#!/usr/bin/env python3

from absl import logging
from absl.testing import absltest

import numpy as np
from sklearn import metrics

from ypack import power

class UnitTests(absltest.TestCase):

    def test_power_method_symmetric(self):
        A = np.random.randn(50, 50)
        A = A + A.T
        ev, v1 = power.power_method_square_matrix(A)
        E, V = np.linalg.eig(A)
        self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([v1], [V[:, 0]])).item(), 1, places=4)

    def test_power_method_nonsquare(self):
        A = np.random.randn(30, 50)
        sv, u1, v1 = power.power_method_nonsquare_matrix(A)
        U, S, Vt = np.linalg.svd(A)
        self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([u1], [U[:, 0]])).item(), 1, places=4)
        self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([v1], [Vt[0, :]])).item(), 1, places=4)




if __name__ == '__main__':
    absltest.main()
