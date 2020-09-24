from .iNode import iNode
import numpy as np


class Sigmoid(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		result = 1 / (1 + np.exp(-inparam))
		return result