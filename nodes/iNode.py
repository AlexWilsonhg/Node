from abc import ABC, abstractmethod
import numpy as np

class iNode(ABC):

	def __init__(self):	
		self.input_slots = []
		self.output_slots = []

	@abstractmethod
	def compute(self):
		pass

