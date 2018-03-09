#!/usr/bin/env python3

# from confprod import confprod
import unittest
from tensorflow import test as tftest
import tensorflow as tf
from tensorflow.python.training.saver import BaseSaverBuilder
import tempfile
import sh
import os
import ypack
import logging
logging.basicConfig(level=logging.INFO)


# class SGD(confprod.Parameterized):
    # pass


# class AdaGrad(confprod.Parameterized):
    # pass


# class L2RegularizedLogisticLoss(confprod.Parameterized):
    # pass


# class RNN(confprod.Parameterized):
    # pass


# conf_specs = [
    # {
        # "optimization": [
            # (SGD, dict(stepSize=[0.5, 1.0, 2.0], sampleSize=10)),
            # (AdaGrad, dict(stepSize=0.1, power=2.0, sampleSize=10)),
        # ],
        # "loss": (L2RegularizedLogisticLoss, dict(amount=0.01))
    # },
    # {
        # "optimization": [
            # (AdaGrad, dict(stepSize=1.0, stabilityConstant=1.0E-9, power=[2.0, 4.0], sampleSize=10)),
        # ]
    # },
# ]

# defaults = {
    # "optimization": {
        # "default": AdaGrad,
        # SGD: dict(stepSize=1.0, sampleSize=1),
        # AdaGrad: dict(stepSize=2.0, stabilityConstant=1e-9, sampleSize=1)
    # },
    # "loss": {
        # L2RegularizedLogisticLoss: dict(amount=1e-3)
    # },
    # "model": {
        # RNN: dict(numNodes=25, sigmoidStrength=0.5)
    # }

# }


# class ConfProdTestCase(unittest.TestCase):
    # def test_empty(self):
        # cs = confprod.generate_configurations([])
        # self.assertEqual(len(cs), 0)

    # def test_generate(self):
        # confs = confprod.generate_configurations(conf_specs)
        # self.assertEqual(len(confs), 6)

    # def test_merge(self):
        # confs = confprod.generate_configurations(conf_specs)
        # for conf in confs:
            # confprod.merge_with_defaults(conf, defaults)
            # self.assertTrue("model" in conf)

    # def test_default(self):
        # conf = confprod.merge_with_defaults(confprod.generate_configurations([{}])[0], defaults)
        # self.assertEqual(conf["optimization"][0], AdaGrad)

    # def test_instantiate(self):
        # conf = confprod.generate_instantiated_configurations([{}], defaults).__next__()
        # self.assertIsInstance(conf["loss"], L2RegularizedLogisticLoss)
        # self.assertAlmostEqual(conf["loss"].get_parameter("amount"), 1e-3)


class SaveRestoreTestCase(tftest.TestCase):

    def _buildGraph(self, size=1, value=1.):
        for i in range(size):
            tf.Variable(value, name='v_' + str(i))

    def _saveAndRestore(self, save_size=1, restore_size=1, restore_size_2=1, saver_class=tf.train.Saver, builder_class=BaseSaverBuilder):
        with tempfile.TemporaryDirectory() as path:
            fn = 'checkpoint.ckpt'
            afn = os.path.join(path, fn)

            with tf.Graph().as_default():
                graph = tf.get_default_graph()
                self._buildGraph(save_size)
                saver = saver_class(builder=builder_class(), filename=afn, max_to_keep=1)
                self.assertEqual(len(tf.global_variables()), save_size)

                with self.test_session(tf.get_default_graph()) as sess:
                    sess.run(tf.global_variables_initializer())
                    tf.get_default_graph().finalize()
                    saver.save(sess, afn)
                    self.assertIn(fn + '.index', os.listdir(path))

            with tf.Graph().as_default():
                graph = tf.get_default_graph()
                self._buildGraph(restore_size)
                saver = saver_class(builder=builder_class(), filename=afn, max_to_keep=1)
                self.assertEqual(len(tf.global_variables()), restore_size)

                with self.test_session(tf.get_default_graph()) as sess:
                    sess.run(tf.global_variables_initializer())
                    tf.get_default_graph().finalize()
                    saver.restore(sess, afn)
                    saver.save(sess, afn)

            with tf.Graph().as_default():
                graph = tf.get_default_graph()
                self._buildGraph(restore_size_2)
                saver = tf.train.Saver(filename=afn, max_to_keep=1)
                self.assertEqual(len(tf.global_variables()), restore_size_2)

                with self.test_session(tf.get_default_graph()) as sess:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, afn)


    def testSaveRestore(self):
        self._saveAndRestore(2, 2)

    def testSaveMoreThanRestore(self):
        self._saveAndRestore(3, 2)

    def testSaveLessThanRestore(self):
        try:
            self._saveAndRestore(2, 3)
            self.fail()
        except tf.errors.NotFoundError:
            pass

    def testDynamicSaveRestore(self):
        self._saveAndRestore(2, 2, saver_class=ypack.OptimisticRestoreSaver)

    def testDynamicSaveMoreThanRestore(self):
        self._saveAndRestore(3, 2, saver_class=ypack.OptimisticRestoreSaver)

    def testDynamicSaveLessThanRestore(self):
        self._saveAndRestore(2, 3, saver_class=ypack.OptimisticRestoreSaver)

    def testDynamicSaveLessThanRestoreThenSaveThenLoadNormal(self):
        self._saveAndRestore(2, 3, 3, saver_class=ypack.OptimisticRestoreSaver)

    def testDynamicSaveLessThanRestoreThenSaveThenLoadMoreNormal(self):
        try:
            self._saveAndRestore(2, 3, 4, saver_class=ypack.OptimisticRestoreSaver)
            self.fail()
        except tf.errors.NotFoundError:
            pass


if __name__ == '__main__':
    # unittest.main()
    tftest.main()
