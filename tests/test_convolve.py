import unittest
from Convolve import Convolve	
import numpy as np

class TestConvolve(unittest.TestCase):

	def test_compute(self):
		convolve = Convolve([[0.1,0],[0,0.1]], keepdims = False)

		indata = np.array([[5,2],[3,2]])
		outdata = convolve.compute(indata)
		self.assertTrue(np.allclose(outdata, 0.7,))

		indata = np.array([[5,2,7,10],[1,8,3,4]])
		outdata = convolve.compute(indata)
		self.assertTrue(np.allclose(outdata, [1.3, 0.5, 1.1]))