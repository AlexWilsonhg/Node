import unittest
from network.Layers import Flatten
import numpy as np

class TestFlatten(unittest.TestCase):

    def test_flatten_5by5_matrix(self):
        flatten = Flatten()

        indata = np.array([[1, 2, 3, 4, 0],
                           [5, 0, 3, 2, 0],
                           [1, 2, 3, 4, 0],
                           [0, 0, 1, 1, 1],
                           [7, 1, 0, 5, 0]])

        outdata = flatten.compute(indata)   
        self.assertTrue(np.allclose(outdata, np.array([1, 2, 3, 4, 0, 5, 0, 3, 
                                                       2, 0, 1, 2, 3, 4, 0, 0, 
                                                       0, 1, 1, 1, 7, 1, 0, 5, 0])))
                                   

    def test_flatten_one_value(self):
        flatten = Flatten()

        indata = 5
        outdata = flatten.compute(indata)
        self.assertEqual(outdata, indata)
       