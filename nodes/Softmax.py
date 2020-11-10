from .iNode import iNode
import numpy as np

class Softmax(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		return np.exp(inparam) / (np.sum(np.exp(inparam) + 0.000000000001))