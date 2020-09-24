from abc import ABC, abstractmethod

class iNode(ABC):

	def __init__(self):	
		self.input_slots = []
		self.output_slots = []

	@abstractmethod
	def compute(self):
		pass

