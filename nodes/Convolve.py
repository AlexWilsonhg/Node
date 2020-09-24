from .iNode import iNode
from scipy import signal

class Convolve(iNode):

	def __init__(self, kernel, stride=1, keepdims=True):
		self.kernel = kernel
		self.stride = stride
		self.keepdims = keepdims

	def compute(self, inparam):
		if not self.keepdims:
			mode = 'valid'
		else:
			mode = 'same'

		result = signal.fftconvolve(inparam, self.kernel, mode = 'valid')
		return result