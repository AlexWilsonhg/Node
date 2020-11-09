import unittest
from nodes.Convolve import Convolve	
import numpy as np

class TestConvolve(unittest.TestCase):

	def test_convolve_with_2by2_kernel(self):
		convolve = Convolve([[0.1,0],[0,0.1]], keepdims = False)

		indata = np.array([[5,2],[3,2]])
		outdata = convolve.compute(indata)
		self.assertTrue(np.allclose(outdata, 0.7,))

		indata = np.array([[5,2,7,10],[1,8,3,4]])
		outdata = convolve.compute(indata)
		self.assertTrue(np.allclose(outdata, [1.3, 0.5, 1.1]))


	def test_convolve_with_3by3_kernel(self):
		convolve = Convolve([[0, 1, 0],
							 [0, 1, 0],
							 [0, 1, 0]], keepdims = False)

		indata = np.array([[1, 0, 1, 1, 0],
						   [0, 1, 1, 1, 1],
						   [1, 1, 0, 1, 1],
						   [0, 1, 1, 1, 0],
						   [1, 1, 0, 1, 1]])

		outdata = convolve.compute(indata)
		print(outdata)
		self.assertTrue(np.allclose(outdata, [[2, 2, 3],
			                                  [3, 2, 3],
			                                  [3, 1, 3]]))
