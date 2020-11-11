import unittest
from network.Layers import Normalize
import numpy as np

class TestNormalize(unittest.TestCase):

    def test_normalize_vector(self):

        norm = Normalize();

        indata = np.array([2,7,3])
        outdata = norm.compute(indata)
        self.assertTrue(np.allclose(outdata, [-0.4, 0.6, -0.2], 1e-10))


    def test_normalize_2d_matrix(self):

        norm = Normalize()

        indata = np.array([[2,1,1],[2,0,1]])
        outdata = norm.compute(indata)
        self.assertTrue(np.allclose(outdata, [[0.4166666666, -0.0833333333, -0.0833333333],
                                              [0.4166666666, -0.5833333333, -0.0833333333]],
                                               1e-6))


    def test_normalize_vector_of_zeros(self):

        norm = Normalize()

        indata = np.array([0,0,0])
        outdata = norm.compute(indata)
        self.assertTrue(np.allclose(outdata, [0,0,0], 1e-10))


    def test_normalize_single_zero(self):

        norm = Normalize()

        indata = np.array([0])
        outdata = norm.compute(indata)
        self.assertTrue(np.allclose(outdata, [0], 1e-10))