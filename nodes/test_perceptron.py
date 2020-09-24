import unittest
from Perceptron import Perceptron
import numpy as np

class TestPerceptron(unittest.TestCase):

	def test_compute(self):
		perceptron = Perceptron(np.array([[0.2,0.1],[0.3,0.5]]), np.array([1,1]))

		indata = np.array([1,5])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.array_equal(outdata, [0.7,2.8]))

		indata = np.array([5,1])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.array_equal(outdata, [1.1,2.0]))

		indata = np.array([-1,-1])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.allclose(outdata, [-0.3,-0.8], 1e-10))

		indata = np.array([0,0])
		outdata = perceptron.compute(indata)
		self.assertTrue(np.allclose(outdata, [0,0], 1e-10))