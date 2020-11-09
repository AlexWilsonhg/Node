from .iNode import iNode
import numpy as np

class LeakyRelu(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):

		return np.where(inparam > 0, inparam, inparam * 0.01)

		