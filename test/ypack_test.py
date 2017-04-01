from confprod import confprod
import unittest


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


if __name__ == '__main__':
    unittest.main()
