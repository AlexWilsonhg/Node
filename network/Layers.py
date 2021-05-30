from abc import ABC, abstractmethod
import skimage.measure
from scipy import signal
from scipy.special import expit
import numpy as np

class iNode(ABC):

	def __init__(self):
		pass

	@abstractmethod
	def compute(self):
		pass


class AveragePool(iNode):

	def __init__(self, stride = 2):
		self.stride = stride

	def compute(self, inparam):
		return skimage.measure.block_reduce(inparam, (2,2), np.average)


class Convolve(iNode):

	def __init__(self, kernel, stride=1, keepdims=True):
		self.kernel = kernel
		self.stride = stride
		self.keepdims = keepdims

	def compute(self, inparam):
		if not self.keepdims:
			_mode = 'valid'
		else:
			_mode = 'same'

		result = signal.fftconvolve(inparam, self.kernel, mode = _mode)
		return result


class Dropout(iNode):

	def __init__(self, dropout_rate):
		self.dropout_rate = dropout_rate


	def compute(self, inparam):
		mask = np.random.choice([0,1], size = inparam.shape,
								p = [self.dropout_rate, 1-self.dropout_rate])

		return inparam * mask


class Flatten(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		return np.array(inparam).flatten()


class LeakyRelu(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):

		return np.where(inparam > 0, inparam, inparam * 0.01)


class MaxPool(iNode):

	def __init__(self, stride = 2):
		self.stride = 2

	def compute(self, inparam):
		return skimage.measure.block_reduce(inparam, (2,2), np.max)


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


class Perceptron(iNode):

	def __init__(self, weights, biases):
		self.weights = weights
		self.biases = biases

	def compute(self, inparam):
		result = np.dot(inparam, self.weights.transpose()) + self.biases
		return result


class Relu(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		return np.maximum(inparam, 0)


class Sigmoid(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		result = expit(inparam)
		return result


class Softmax(iNode):

	def __init__(self):
		pass

	def compute(self, inparam):
		return np.exp(inparam) / (np.sum(np.exp(inparam) + 0.000000000001))
