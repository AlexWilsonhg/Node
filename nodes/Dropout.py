from .iNode import iNode
import numpy as np

class Dropout(iNode):

	def __init__(self, dropout_rate):
		self.dropout_rate = dropout_rate


	def compute(self, inparam):
		mask = np.random.choice([0,1], size = inparam.shape, 
								p = [self.dropout_rate, 1-self.dropout_rate])

		return inparam * mask