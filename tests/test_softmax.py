import unittest
from network.Layers import Softmax
import numpy as np

class TestSoftmax(unittest.TestCase):

	def test_single_number_sums_to_1(self):
		soft = Softmax()

		indata = 5
		outdata = soft.compute(indata)
		self.assertTrue(np.allclose(outdata, 1))

	def test_multiple_positive_numbers_sums_to_1(self):
		soft = Softmax()

		indata = np.array([5,1,2,10])
		outdata = soft.compute(indata)
		self.assertTrue(np.allclose(np.sum(outdata),1))

	def test_positive_numbers_returns_correct_probabilities(self):
		soft = Softmax()

		indata = np.array([0.1,0.2,0.3])
		outdata = soft.compute(indata)
		self.assertTrue(np.allclose(outdata, np.array([0.30061, 0.332225, 0.367165])))

	def test_negative_numbers_returns_correct_probabilities(self):
		soft = Softmax()

		indata = np.array([-0.1,-0.2,-0.3])
		outdata = soft.compute(indata)
		self.assertTrue(np.allclose(outdata, np.array([0.367165, 0.332225, 0.30061])))
