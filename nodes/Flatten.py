from .iNode import iNode
import numpy as np

class Flatten(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		return np.array(inparam).flatten()