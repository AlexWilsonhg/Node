import unittest
from network.Layers import Perceptron
import numpy as np

class TestPerceptron(unittest.TestCase):

	def test_compute(self):
		perceptron = Perceptron(np.array([[0.2,0.1],[0.3,0.5]]), np.array([1,1]))

		indata = np.array([1,5])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.array_equal(outdata, [1.7,3.8]))

		indata = np.array([5,1])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.array_equal(outdata, [2.1,3.0]))

		indata = np.array([-1,-1])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.allclose(outdata, [0.7,0.2], 1e-10))

		indata = np.array([0,0])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.allclose(outdata, [1,1], 1e-10))