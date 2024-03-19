# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, sampler
import math
import numpy as np
from utils.insert_noise import DatasetSplit_flip
from nets.models import get_model

class Synthetic_clients(object):

	def __init__(self, conf, eval_dataset, dict_clients, val_indices, client_id, label):

		self.conf = conf
		self.local_model = get_model(self.conf["model_name"], num_classes=self.conf['num_classes'])
		self.client_id = client_id
		self.eval_dataset = eval_dataset
		self.val_indices = val_indices
		self.dict_clients = dict_clients


		if label == '0':
			self.noise_rate = 1  # synthetic noisy client
			self.client_label = "synthetic noisy"
		elif label == '1':
			self.noise_rate = 0  # synthetic clean client
			self.client_label = "synthetic clean"

		self.noisy_samples_id = np.random.choice(
			int(len(self.val_indices) / (2*self.conf['num_pairs'])),
			math.floor(len(self.val_indices) / (2*self.conf['num_pairs']) * self.noise_rate),
			replace=False)

		# random flipping noise
		self.train_loader = DataLoader(DatasetSplit_flip(self.conf, self.eval_dataset,
		                                                 self.dict_clients[self.client_id], self.noisy_samples_id),
		                               batch_size=10, shuffle=True)

		self.val_loader = DataLoader(self.eval_dataset, batch_size=10, sampler=sampler.SubsetRandomSampler(self.val_indices))


	def local_train(self, model):
		local_data_length = len(self.dict_clients[self.client_id])
		original_labels = np.array(self.eval_dataset.targets)[list(self.dict_clients[self.client_id])].tolist()
		print('Original data labels:', set(original_labels))
		for item in set(original_labels):
			print('the %d has found %d' % (item, original_labels.count(item)))

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())

		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=0.9)
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda().float()
					target = target.cuda()

				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
				optimizer.step()

		# print("Local Epoch %d done." % self.conf["local_epochs"])
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])

		# validate local model
		correct, total, loss = 0, 0, 0.0
		with torch.no_grad():
			for batch_id, batch in enumerate(self.val_loader):
				data, targets = batch

				if torch.cuda.is_available():
					data = data.cuda()
					targets = targets.cuda()

				outputs = self.local_model(data)
				loss += torch.nn.functional.cross_entropy(outputs, targets,
											  reduction='sum').item()
				predicted = outputs.data.max(1)[1]
				total += targets.size(0)
				correct += (predicted == targets).sum().item()

		val_acc = correct / total
		val_loss = loss / total

		return diff, local_data_length, val_acc, val_loss, self.client_label
