import unittest
import numpy as np
from network.Network import Network
from network.Layers import *

class TestNetwork(unittest.TestCase):

    def test_single_layer_network(self):

        network = Network()

        network.add_layer(Normalize())

        indata = np.array([2, 7, 3])
        outdata = network.compute(indata)
        self.assertTrue(np.allclose(outdata, [-0.4, 0.6, -0.2]))

    def test_multi_layer_network(self):

        network = Network()
        network.add_layer(Convolve([[0.1, 0],
                                    [0, 0.1]], keepdims = False))
        network.add_layer(Relu())
        network.add_layer(Flatten())
        network.add_layer(Softmax())

        indata = np.array([[5, 2, 7, 10],
                           [1, 8, 3,  4]])

        outdata = network.compute(indata)

        self.assertTrue(np.allclose(outdata, [0.440905, 0.19811, 0.36098]))