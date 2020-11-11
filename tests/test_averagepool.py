import unittest
from network.Layers import AveragePool
import numpy as np

class TestAveragePool(unittest.TestCase):

	def test_2d_average_pool_positive_numbers(self):
		pool = AveragePool()

		indata = np.array([[5, 1],[8, 0]])
		outdata = pool.compute(indata)
		self.assertEqual(outdata, 3.5)
