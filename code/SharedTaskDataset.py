#!/usr/bin/env python

"""
Author: Peratham Wiriyathammabhum


"""
import os
import json

from torch.utils.data import Dataset, DataLoader

class SharedTaskDataset(Dataset):
	"""PyTorch Dataset class for loading data.

	  This is where the data parsing happens.

	  This class is built with reusability in mind: it can be used as is as.

	  Arguments:

		path (:obj:`str`):
			Path to the data partition.

	 """

	def __init__(self, path, split):
		# Check if path exists.
		if not os.path.isdir(path):
			# Raise error if path is invalid.
			raise ValueError('Invalid `path` variable! Needs to be a directory')

		self.texts = []
		self.labels = []

		with open('{}/{}_concat.json'.format(path, split),'r') as f:
			data = json.load(f)

		self.texts = data['data']
		self.labels = data['labels']

		# Number of exmaples.
		self.n_examples = len(self.labels)

		return

	def __len__(self):
		"""When used `len` return the number of examples.

		"""
		return self.n_examples

	def __getitem__(self, idx):
		"""Given an index return an example from the position.
		
			Arguments:

			  item (:obj:`int`):
				  Index position to pick an example to return.

			Returns:
			  :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
			  asociated labels.

		"""
		# idx = item % self.__len__()

		return {'text':self.texts[idx], 'label':self.labels[idx]}
