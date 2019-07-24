#!/usr/bin/env python3

from absl import logging
from absl.testing import absltest

from ypack import th_utils
import torch as th
import numpy as np
from sklearn import metrics



class UnitTests(absltest.TestCase):

    def test_iterative_svd(self):
        A = np.random.randn(30, 50)
        A_th = th.from_numpy(A)


        def fn(x):
            return th.mm(x, A_th.t())

        u_list, v_list = th_utils.general_iterative_svd(fn, th.ones_like(A_th[0]).unsqueeze_(0), fb_iters=1000)
        u1, v1 = u_list[0][0].numpy(), v_list[0][0].numpy()

        U, S, Vt = np.linalg.svd(A)
        self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([u1], [U[:, 0]])).item(), 1, places=4)
        self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([v1], [Vt[0, :]])).item(), 1, places=4)

    def test_network_linear(self):
        A = np.random.randn(30, 50)
        A_th = th.from_numpy(A)

        def fn(x, i):
            return x, th.mm(x, A_th.t())

        for bias in (True, False):
            u_list, s_list, v_list = th_utils.network_linear(fn, th.ones_like(A_th[:2]), num_vectors=3, with_bias=bias, num_noise_samples=128)
            u1, v1 = u_list[0][0].numpy(), v_list[0][0].numpy()

            U, S, Vt = np.linalg.svd(A)
            self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([u1], [U[:, 0]])).item(), 1, places=4)
            self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([v1], [Vt[0, :]])).item(), 1, places=4)

    def test_neural_net(self):
        net = th.nn.Sequential(
            th.nn.Linear(20, 30),
            th.nn.ReLU(),
            th.nn.Linear(30, 10)
        )
        x0 = th.ones(20).unsqueeze_(0)

        def fn_it(x):
            return net(x)

        def fn(x, i):
            return x, net(x)

        u_list_it, v_list_it = th_utils.general_iterative_svd(fn_it, x0, fb_iters=10000)
        u1_it, v1_it = u_list_it[0][0].numpy(), v_list_it[0][0].numpy()

        for bias in (False, True):
            print('bias: {}'.format(bias))
            u_list, s_list, v_list = th_utils.network_linear(fn, x0, num_vectors=3, with_bias=bias, num_noise_samples=100000)
            u1, v1 = u_list[0][0].numpy(), v_list[0][0].numpy()

            self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([u1], [u1_it])).item(), 1, places=4)
            self.assertAlmostEqual(np.abs(metrics.pairwise.cosine_similarity([v1], [v1_it])).item(), 1, places=4)



if __name__ == '__main__':
    absltest.main()
