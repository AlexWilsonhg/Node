import unittest
from network.Layers import MaxPool
import numpy as np

class TestMaxPool(unittest.TestCase):

	def test_2d_max_pool_positive_numbers(self):
		pool = MaxPool()

		indata = np.array([[10, 5, 1, 10],
			              [ 5, 1, 8, 10]])
		outdata = pool.compute(indata)
		self.assertTrue(np.allclose(outdata, np.array([10, 10])))


	def test_2d_max_pool_negative_numbers(self):
		pool = MaxPool()

		indata = np.array([[-5, -20, -1, 0],
			               [-25,  0,  0,-5]])
		outdata = pool.compute(indata)
		self.assertTrue(np.allclose(outdata, np.array([0, 0])))