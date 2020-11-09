from .iNode import iNode
import numpy as np

class Relu(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		return np.maximum(inparam, 0)
