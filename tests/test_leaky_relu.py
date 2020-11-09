import unittest
import numpy as np
from nodes.LeakyRelu import LeakyRelu

class TestLeakyRelu(unittest.TestCase):

	def test_leaky_relu(self):

		lRelu = LeakyRelu()

		indata = np.array([0,0])
		outdata = lRelu.compute(indata)
		self.assertTrue(np.allclose(outdata, [0,0]))

		indata = np.array([-5,5])
		outdata = lRelu.compute(indata)
		self.assertTrue(np.allclose(outdata, [-0.05, 5]))

		indata = np.array([-50000,50000])
		outdata = lRelu.compute(indata)
		self.assertTrue(np.allclose(outdata, [-500.0, 50000]))