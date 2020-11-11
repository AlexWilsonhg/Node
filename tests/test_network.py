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
