import unittest
import numpy as np
from nodes.Relu import Relu


class TestRelu(unittest.TestCase):

	def test_relu(self):

		relu = Relu()

		indata = np.array([0,1,1])
		outdata = relu.compute(indata)
		self.assertTrue(np.array_equal(outdata, [0,1,1]))

		indata = np.array([-1,0])
		outdata = relu.compute(indata)
		self.assertTrue(np.array_equal(outdata, [0,0]))

		indata = np.array([5000,-5000])
		outdata = relu.compute(indata)
		self.assertTrue(np.array_equal(outdata, [5000,0]))