import unittest
from nodes.Sigmoid import Sigmoid
import numpy as np

class TestSigmoid(unittest.TestCase):

	def test_compute(self):
		sigmoid = Sigmoid()

		indata = np.array([0.5,0.1,2])
		outdata = sigmoid.compute(indata)
		self.assertTrue(np.allclose(outdata, [0.622459331, 0.524979187, 0.880797078], 1e-10))

		indata = np.array([0,0,0])
		outdata = sigmoid.compute(indata)
		self.assertTrue(np.allclose(outdata, [0.5, 0.5, 0.5], 1e-10))

		indata = np.array([10000, 50000, 8000])
		outdata = sigmoid.compute(indata)
		self.assertTrue(np.allclose(outdata, [1,1,1], 1e-10))

		indata = np.array([-10000, -5000, -8000])
		outdata	= sigmoid.compute(indata)
		self.assertTrue(np.allclose(outdata, [0,0,0], 1e-10))