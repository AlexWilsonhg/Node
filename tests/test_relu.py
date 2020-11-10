import unittest
import numpy as np
from nodes.Relu import Relu


class TestRelu(unittest.TestCase):

	def test_single_value(self):

		relu = Relu()

		indata = 1
		outdata = relu.compute(indata)
		self.assertTrue(np.array_equal(outdata, 1))

	def test_vector(self):

		relu = Relu()

		indata = np.array([-1,0,500])
		outdata = relu.compute(indata)
		self.assertTrue(np.array_equal(outdata, [0,0,500]))

	def test_2d_matrix(self):

		relu = Relu()

		indata = np.array([[5000,-5000],
						   [0   ,    0],
						   [-100,  500]])

		outdata = relu.compute(indata)
		self.assertTrue(np.array_equal(outdata, np.array([[5000,	0],
														  [0,	    0],
														  [0,	  500]])))