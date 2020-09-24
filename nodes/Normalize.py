from .iNode import iNode
import numpy as np

class Normalize(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):

		if(np.all(inparam == 0)):
			return inparam

		valueRange = np.max(inparam) - np.min(inparam)
		valueMean = np.mean(inparam)

		result = []
		for i in inparam:
			result.append((i-valueMean) / valueRange)
		return result