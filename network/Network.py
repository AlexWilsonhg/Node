from .Layers import *

class Network():

	def __init__(self):
		self.layers = []

	def compute(self, inparam):

		data = inparam

		for layer in self.layers:
			data = layer.compute(data)


		return data

	def add_layer(self, layer):
		self.layers.append(layer)