# -*- coding: utf-8 -*-
"""
@Time: 2022/3/15 7:24 PM
@Author: Ran Zhu
@Email: ranzhuzr@gmail.com
@IDE: PyCharm
"""

import math
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, sampler


# ======================== label noise ======================= #
# random flipping noise
class DatasetSplit_flip(Dataset):
    def __init__(self, conf, dataset, idxs, samples_flip_id):
        self.conf = conf
        self.dataset = dataset
        self.idxs = list(idxs)
        self.samples_flip_id = samples_flip_id

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if item in self.samples_flip_id:
            label_1 = label
            while label_1 == label:
               label_1 = random.randrange(self.conf["num_classes"])
            label = label_1
        return image, label

# pair flipping noise
class DatasetSplit_flip_pair_noise(Dataset):

    def __init__(self, conf, dataset, idxs, samples_flip_id):
        self.conf = conf
        self.dataset = dataset
        self.idxs = list(idxs)
        self.samples_flip_id = samples_flip_id
        # assign structure label noise by assigning labels to the most often confused classes,
        # as determined by a confusion matrix from a centralized training
        if self.conf['data'] == 'cifar10':
            self.structure = [8, 9, 4, 5, 3, 3, 3, 5, 0, 1]
        elif self.conf['data'] == 'fashionmnist':
            self.structure = [6, 6, 6, 6, 6, 8, 4, 5, 8, 8]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if item in self.samples_flip_id:
            label = self.structure[label]

        return image, label

# weighted flipping noise
def label_of_certain_prob(seq, num_list):
    prob = []
    for num in num_list:
        prob.append(num/sum(num_list))
    x = random.uniform(0, 1)
    cumulative_prob = 0.0
    for item, item_prob in zip(seq, prob):
        cumulative_prob += item_prob
        if x <= cumulative_prob:
            break
    return item

class DatasetSplit_flip_weighted_noise_(Dataset):
    def __init__(self, conf, dataset, idxs, samples_flip_id):
        self.conf = conf
        self.dataset = dataset
        self.idxs = list(idxs)
        self.samples_flip_id = samples_flip_id
        if self.conf['data'] == 'cifar10':
            self.seq_dict ={"0": [1, 2, 3, 4, 5, 6, 7, 8, 9],
	                        "1": [0, 2, 3, 4, 5, 6, 7, 8, 9],
	                        "2": [0, 1, 3, 4, 5, 6, 7, 8, 9],
	                        "3": [0, 1, 2, 4, 5, 6, 7, 8, 9],
	                        "4": [0, 1, 2, 3, 5, 6, 7, 8, 9],
	                        "5": [0, 1, 2, 3, 4, 6, 7, 8, 9],
	                        "6": [0, 1, 2, 3, 4, 5, 7, 8, 9],
	                        "7": [0, 1, 2, 3, 4, 5, 6, 8, 9],
	                        "8": [0, 1, 2, 3, 4, 5, 6, 7, 9],
	                        "9": [0, 1, 2, 3, 4, 5, 6, 7, 8]
	                        }
            self.structure_dict = {"0": [12, 26, 6, 10, 3, 3, 4, 29, 16],
	                               "1": [7, 3, 4, 0, 1, 1, 0, 25, 47],
	                               "2": [44, 4, 45, 50, 31, 26, 6, 2, 2],
	                               "3": [14, 9, 40, 43, 124, 29, 19, 8, 10],
	                               "4": [16, 2, 36, 40, 17, 17, 25, 5, 1],
	                               "5": [12, 4, 23, 130, 33, 17, 30, 1, 9],
	                               "6": [7, 3, 20, 36, 21, 13, 2, 2, 2],
	                               "7": [15, 2, 13, 23, 32, 36, 0, 0, 4],
	                               "8": [46, 7, 4, 7, 0, 4, 1, 2, 11],
	                               "9": [18, 32, 1, 5, 1, 0, 3, 2, 26]
	                               }
        else:
            exit('Error: weighted flipping noise only works in cifar10 dataset.')
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if item in self.samples_flip_id:
            label = label_of_certain_prob(self.seq_dict["{}".format(label)], self.structure_dict["{}".format(label)])

        return image, label

# openset noise
class DatasetSplit_flip_openset(Dataset):
    def __init__(self, conf, dataset, openset_dataset, idxs, openset_idxs, client_id, samples_flip_id):
        self.conf = conf
        self.dataset = dataset
        self.openset_dataset = openset_dataset
        self.idxs = list(idxs)
        self.openset_idxs = list(openset_idxs)
        self.client_id = client_id
        self.samples_flip_id = samples_flip_id

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if item in self.samples_flip_id:
            image, label = self.openset_dataset[self.openset_idxs[item]] # else(for Oov_type=v2)
            if label > 9:
                label = random.randrange(self.conf["num_classes"])

        return image, label




# ======================= feature noise ====================== #
class AddGaussianNoise():
    def __init__(self, mean=0.2, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        c, h, w = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(1, h, w))
        N = np.repeat(N, c, axis=0)
        img = N + np.array(img)
        img[img > 1] = 1.0
        img[img < 0] = 0.0

        return img

# feature noise including Gaussian noise, Resolution noise, and Corruption noise
class DatasetSplit_feature_noise(Dataset):
    def __init__(self, conf, feature_noise_type, dataset, idxs, client_id, samples_flip_id):
        self.conf = conf
        self.feature_noise_type = feature_noise_type
        self.dataset = dataset
        self.idxs = list(idxs)
        self.client_id = client_id
        self.samples_flip_id = samples_flip_id
        if self.feature_noise_type == "GN":
            self.transform = transforms.Compose([AddGaussianNoise()])

        elif self.feature_noise_type == "RC":
            if self.conf['data'] == "cifar10":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(size=(32, 32), scale=(0.015625, 0.015625), ratio=(1.0, 1.0))])
            elif self.conf['data'] == "fashionmnist":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(size=(28, 28), scale=(0.015625, 0.015625), ratio=(1.0, 1.0))])

        elif self.feature_noise_type == "CP":
            if self.conf['data'] == "cifar10":
                self.transform = transforms.Compose([transforms.RandomErasing(p=1, scale=(0.5, 0.5), ratio=(1.0, 1.0), value=(0, 0, 0))])
            elif self.conf['data'] == "fashionmnist":
                self.transform = transforms.Compose([transforms.RandomErasing()])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if item in self.samples_flip_id:
            if self.feature_noise_type == "GN":
                image = torch.from_numpy(self.transform(image))
            else:
                image = self.transform(image)

        return image, label



# ====================== noise insertion ===================== #

def Dataset_noise_hybrid_across(conf, train_dataset, dict_clients, data_for_openset_noise,
                                dict_clients_for_open_noise, noisy_samples_id, client_id):
	num_segment = int(math.ceil(conf['num_models'] * conf['noisy_client_rate'] / 6))
	# random flipping
	if int(client_id) < num_segment:
		print("Client {} has random flipping noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_flip(conf, train_dataset, dict_clients[client_id], samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# pair flipping
	elif num_segment <= int(client_id) < 2 * num_segment:
		print("Client {} has pair flipping noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_flip_pair_noise(conf, train_dataset,dict_clients[client_id], samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# Open-set noise
	elif 2 * num_segment <= int(client_id) < 3 * num_segment:
		print("Client {} has open-set noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_flip_openset(conf, train_dataset, data_for_openset_noise, dict_clients[client_id],
			                      dict_clients_for_open_noise[client_id], client_id=client_id,
			                      samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# Gaussian noise
	elif 3 * num_segment <= int(client_id) < 4 * num_segment:
		print("Client {} has feature noise---Gaussian Noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_feature_noise(conf, "GN", train_dataset, dict_clients[client_id], client_id=client_id,
			                           samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# Resolution noise
	elif 4 * num_segment <= int(client_id) < 5 * num_segment:
		print("Client {} has feature noise---Resize Crop".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_feature_noise(conf, "RC", train_dataset, dict_clients[client_id], client_id=client_id,
			                           samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# Corruption noise
	elif 5 * num_segment <= int(client_id) < 6 * num_segment:
		print("Client {} has feature noise---Corruption".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_feature_noise(conf, "CP", train_dataset, dict_clients[client_id], client_id=client_id,
			                           samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# clean clients
	else:
		print("Client {} has no noise".format(client_id))
		train_loader = DataLoader(dataset=train_dataset, sampler=sampler.SubsetRandomSampler(dict_clients[client_id]),
		                          batch_size=conf['batch_size'], drop_last=True)

	return train_loader

def Dataset_noise_label_across(conf, train_dataset, dict_clients, data_for_openset_noise,
                                dict_clients_for_open_noise, noisy_samples_id, client_id):
	num_segment = int(math.ceil(conf['num_models'] * conf['noisy_client_rate'] / 3))
	# random flipping
	if int(client_id) < num_segment:
		print("Client {} has random flipping noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_flip(conf, train_dataset, dict_clients[client_id], samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# pair flipping
	elif num_segment <= int(client_id) < 2 * num_segment:
		print("Client {} has pair flipping noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_flip_pair_noise(conf, train_dataset,dict_clients[client_id], samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# Open-set noise
	elif 2 * num_segment <= int(client_id) < 3 * num_segment:
		print("Client {} has open-set noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_flip_openset(conf, train_dataset, data_for_openset_noise, dict_clients[client_id],
			                      dict_clients_for_open_noise[client_id], client_id=client_id,
			                      samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# clean clients
	else:
		print("Client {} has no noise".format(client_id))
		train_loader = DataLoader(dataset=train_dataset, sampler=sampler.SubsetRandomSampler(dict_clients[client_id]),
		                          batch_size=conf['batch_size'], drop_last=True)

	return train_loader

def Dataset_noise_feature_across(conf, train_dataset, dict_clients, noisy_samples_id, client_id):
	num_segment = int(math.ceil(conf['num_models'] * conf['noisy_client_rate'] / 3))
	# Gaussian noise
	if int(client_id) < num_segment:
		print("Client {} has feature noise---Gaussian Noise".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_feature_noise(conf, "GN", train_dataset, dict_clients[client_id], client_id=client_id,
			                           samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# Resolution noise
	elif num_segment <= int(client_id) < 2 * num_segment:
		print("Client {} has feature noise---Resize Crop".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_feature_noise(conf, "RC", train_dataset, dict_clients[client_id], client_id=client_id,
			                           samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# Corruption noise
	elif 2 * num_segment <= int(client_id) < 3 * num_segment:
		print("Client {} has feature noise---Corruption".format(client_id))
		train_loader = DataLoader(
			DatasetSplit_feature_noise(conf, "CP", train_dataset, dict_clients[client_id], client_id=client_id,
			                           samples_flip_id=noisy_samples_id),
			batch_size=conf['batch_size'], shuffle=True, drop_last=True)
	# clean clients
	else:
		print("Client {} has no noise".format(client_id))
		train_loader = DataLoader(dataset=train_dataset, sampler=sampler.SubsetRandomSampler(dict_clients[client_id]),
		                          batch_size=conf['batch_size'], drop_last=True)

	return train_loader

class Dataset_noise_hybrid_intra(Dataset):
	def __init__(self, conf, dataset, idxs, clients_data_for_openset_noise, client_idxs_for_open_noise, noisy_samples_id):
		self.conf = conf
		self.dataset = dataset
		self.clients_data_for_openset_noise = clients_data_for_openset_noise
		self.idxs = list(idxs)
		self.client_idxs_for_open_noise = list(client_idxs_for_open_noise)
		self.noisy_samples_id = noisy_samples_id

		self.transform_GN = transforms.Compose([AddGaussianNoise()])
		if self.conf['data'] == "cifar10":
			self.structure = [8, 9, 4, 5, 3, 3, 3, 5, 0, 1]
			self.transform_RC = transforms.Compose([transforms.RandomResizedCrop(size=(32, 32), scale=(0.015625, 0.015625), ratio=(1.0, 1.0))])
			self.transform_CP = transforms.Compose([transforms.RandomErasing(p=1, scale=(0.5, 0.5), ratio=(1.0, 1.0), value=(0, 0, 0))])
		elif self.conf['data'] == "fashionmnist":
			self.structure = [6, 6, 6, 6, 6, 8, 4, 5, 8, 8]
			self.transform_RC = transforms.Compose([transforms.RandomResizedCrop(size=(28, 28), scale=(0.015625, 0.015625), ratio=(1.0, 1.0))])
			self.transform_CP = transforms.Compose([transforms.RandomErasing()])

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self, item):
		image, label = self.dataset[self.idxs[item]]
		# open-set noise
		if item in self.noisy_samples_id[:int(len(self.noisy_samples_id)/6)]:
			image, label = self.clients_data_for_openset_noise[self.client_idxs_for_open_noise[item]]
			if label > 9:
				label = random.randrange(self.conf["num_classes"])
		# random flipping
		elif item in self.noisy_samples_id[int(len(self.noisy_samples_id)/6): int(len(self.noisy_samples_id)/3)]:
			label_1 = label
			while label_1 == label:
				label_1 = random.randrange(self.conf["num_classes"])
			label = label_1
		# pair flipping
		elif item in self.noisy_samples_id[int(len(self.noisy_samples_id)/3): int(len(self.noisy_samples_id)/2)]:
			label = self.structure[label]
		# gaussian noise
		elif item in self.noisy_samples_id[int(len(self.noisy_samples_id)/2): int(2*len(self.noisy_samples_id)/3)]:
			image = torch.from_numpy(self.transform_GN(image))
		# resolution noise
		elif item in self.noisy_samples_id[int(2*len(self.noisy_samples_id)/3): int(5*len(self.noisy_samples_id)/6)]:
			image = self.transform_RC(image)
		# corruption noise
		elif item in self.noisy_samples_id[int(5*len(self.noisy_samples_id)/6): int(len(self.noisy_samples_id))]:
			image = self.transform_CP(image)

		return image, label

class Dataset_noise_label_intra(Dataset):
	def __init__(self, conf, dataset, idxs, clients_data_for_openset_noise, client_idxs_for_open_noise, noisy_samples_id):
		self.conf = conf
		self.dataset = dataset
		self.clients_data_for_openset_noise = clients_data_for_openset_noise
		self.idxs = list(idxs)
		self.client_idxs_for_open_noise = list(client_idxs_for_open_noise)
		self.noisy_samples_id = noisy_samples_id

		if self.conf['data'] == 'cifar10':
			self.structure = [8, 9, 4, 5, 3, 3, 3, 5, 0, 1]
		elif self.conf['data'] == 'fashionmnist':
			self.structure = [6, 6, 6, 6, 6, 8, 4, 5, 8, 8]

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self, item):
		image, label = self.dataset[self.idxs[item]]
		# open-set noise
		if item in self.noisy_samples_id[:int(len(self.noisy_samples_id)/3)]:
			image, label = self.clients_data_for_openset_noise[self.client_idxs_for_open_noise[item]]
			if label > 9:
				label = random.randrange(self.conf["num_classes"])
		# random flipping
		elif item in self.noisy_samples_id[int(len(self.noisy_samples_id)/3): int(2*len(self.noisy_samples_id)/3)]:
			label_1 = label
			while label_1 == label:
				label_1 = random.randrange(self.conf["num_classes"])
			label = label_1
		# pair flipping
		elif item in self.noisy_samples_id[int(2*len(self.noisy_samples_id)/3): int(len(self.noisy_samples_id))]:
			label = self.structure[label]

		return image, label

class Dataset_noise_feature_intra(Dataset):
	def __init__(self, conf, dataset, idxs, noisy_samples_id):
		self.conf = conf
		self.dataset = dataset
		self.idxs = list(idxs)
		self.noisy_samples_id = noisy_samples_id

		self.transform_GN = transforms.Compose([AddGaussianNoise()])
		if self.conf['data'] == "cifar10":
			self.transform_RC = transforms.Compose([transforms.RandomResizedCrop(size=(32, 32), scale=(0.015625, 0.015625), ratio=(1.0, 1.0))])
			self.transform_CP = transforms.Compose([transforms.RandomErasing(p=1, scale=(0.5, 0.5), ratio=(1.0, 1.0), value=(0, 0, 0))])
		elif self.conf['data'] == "fashionmnist":
			self.transform_RC = transforms.Compose([transforms.RandomResizedCrop(size=(28, 28), scale=(0.015625, 0.015625), ratio=(1.0, 1.0))])
			self.transform_CP = transforms.Compose([transforms.RandomErasing()])

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self, item):
		image, label = self.dataset[self.idxs[item]]
		# gaussian noise
		if item in self.noisy_samples_id[:int(len(self.noisy_samples_id)/3)]:
			image = torch.from_numpy(self.transform_GN(image))
		# resolution noise
		elif item in self.noisy_samples_id[int(len(self.noisy_samples_id)/3): int(2*len(self.noisy_samples_id)/3)]:
			image = self.transform_RC(image)
		# corruption noise
		elif item in self.noisy_samples_id[int(2*len(self.noisy_samples_id)/3): int(len(self.noisy_samples_id))]:
			image = self.transform_CP(image)

		return image, label