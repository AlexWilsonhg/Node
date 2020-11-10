import unittest
from nodes.Dropout import Dropout
import numpy as np

class TestDropout(unittest.TestCase):

	def test_5by5_50_dropout(self):
		dropout = Dropout(0.5)

		indata = np.array([[1, 2, 3, 4, 5],
					       [5, 4, 3, 2, 1],
					       [1, 2, 3, 4, 5],
					       [1, 1, 1, 1, 1],
					       [7, 1, 4, 5, 2]])
		np.random.seed(0)
		outdata = dropout.compute(indata)
		self.assertTrue(np.allclose(outdata, np.array([[1, 2, 3, 4, 0],
       												 [5, 0, 3, 2, 0],
                                                     [1, 2, 3, 4, 0],
                                                     [0, 0, 1, 1, 1],
                                                     [7, 1, 0, 5, 0]])))

