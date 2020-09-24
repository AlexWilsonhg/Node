from .iNode import iNode
import numpy as np


class Perceptron(iNode):

	def __init__(self, weights, biases):
		self.weights = weights
		self.biases = biases

	def compute(self, inparam):
		result = np.dot(inparam, self.weights.transpose()) + self.biases
		return result