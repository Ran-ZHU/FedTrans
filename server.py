# -*- coding: utf-8 -*-
import torch
from torch import nn
from discriminator.nn_em import nn_em
import numpy as np
from scipy.special import digamma
from torch.utils.data import DataLoader, sampler, Dataset
from numpy import linalg as LA
from time import time
import pickle
import tensorflow as tf
from copy import deepcopy
from model.models import get_model

class Server(object):
	def __init__(self, conf, eval_dataset, auxiliary_data_indices, test_indices):
		self.conf = conf
		self.nn_em = nn_em()
		self.auxiliary_data_indices = auxiliary_data_indices
		self.eval_dataset = eval_dataset
		self.test_indices = test_indices
		self.hist = None

		self.global_model = get_model(self.conf["model_name"], num_classes=self.conf['num_classes'])

		self.val_loader = DataLoader(self.eval_dataset, batch_size=self.conf["batch_size"],
		                             sampler=sampler.SubsetRandomSampler(self.auxiliary_data_indices))

		self.test_loader = DataLoader(self.eval_dataset, batch_size=self.conf["batch_size"],
		                              sampler=sampler.SubsetRandomSampler(self.test_indices))

	def init_probabilities(self, n_clients):
		qz1 = 0.5 * np.ones((n_clients, 1))
		A = 2
		B = 2

		return 1 - qz1, qz1, A, B

	def init_alpha_beta(self, A, B, n_rounds):
		alpha = np.zeros((n_rounds, 1), dtype='float32')
		beta = np.zeros((n_rounds, 1), dtype='float32')
		for w in range(0, n_rounds):
			alpha[w] = A
			beta[w] = B
		return alpha, beta

	def update(self, a, b, n_update, change):
		n_update += 1
		change += np.abs(a-b)

		return n_update, change

	def e_step(self, n_rounds, n_clients, q_z_i_0, q_z_i_1, sub_round_reputation, alpha, beta, theta_i, y_train, max_it):
		counter = 0
		for it in range(max_it):
			counter += 1
			change = 0
			n_update = 0
			# update q(z)
			for client in range(n_clients):
				updated_q_z_i_0 = (1 - theta_i[client])
				updated_q_z_i_1 = theta_i[client]

				client_i = [k for k in range(n_rounds) if sub_round_reputation[:,client][k] == 1]
				for round in client_i:
					alpha_val = alpha[round]
					beta_val = beta[round]
					updated_q_z_i_0 = updated_q_z_i_0 * np.exp(digamma(beta_val) - digamma(alpha_val + beta_val))
					updated_q_z_i_1 = updated_q_z_i_1 * np.exp(digamma(alpha_val) - digamma(alpha_val + beta_val))

				client_i_n = [k for k in range(n_rounds) if sub_round_reputation[:,client][k] == 0]
				for round in client_i_n:
					alpha_val = alpha[round]
					beta_val = beta[round]
					updated_q_z_i_0 = updated_q_z_i_0 * np.exp(digamma(alpha_val) - digamma(alpha_val + beta_val))
					updated_q_z_i_1 = updated_q_z_i_1 * np.exp(digamma(beta_val) - digamma(alpha_val + beta_val))

				# print (updated_q_z_i_0, updated_q_z_i_1)
				new_q_z_i_1 = updated_q_z_i_1 * 1.0 / (updated_q_z_i_0 + updated_q_z_i_1)
				n_update, change = self.update(q_z_i_1[client], new_q_z_i_1, n_update, change)
				q_z_i_0[client] = updated_q_z_i_0 * 1 / (updated_q_z_i_0 + updated_q_z_i_1)
				q_z_i_1[client] = updated_q_z_i_1 * 1 / (updated_q_z_i_0 + updated_q_z_i_1)
			
			if y_train == 0:
				q_z_i_1_ = q_z_i_1
				q_z_i_0_ = q_z_i_0
			else:
				q_z_i_1_ = np.concatenate((y_train, q_z_i_1[len(y_train):]))
				q_z_i_0_ = np.concatenate((1 - y_train, q_z_i_0[len(y_train):]))

			# update q(r)
			new_alpha = np.zeros((n_rounds, 1))
			new_beta = np.zeros((n_rounds, 1))

			for round in range(0, n_rounds):
				new_alpha[round] = alpha[round]
				new_beta[round] = beta[round]

			for round in range(0, n_rounds):
				round_i = [k for k in range(n_clients) if sub_round_reputation[round,:][k] == 1]
				for client in round_i:
					new_alpha[round] += q_z_i_1_[client]
					new_beta[round] += 1 - q_z_i_1_[client]

				round_i_n = [k for k in range(n_clients) if sub_round_reputation[round,:][k] == 0]
				for client in round_i_n:
					new_alpha[round] += (1 - q_z_i_1_[client])
					new_beta[round] += q_z_i_1_[client]

			for round in range(0, n_rounds):
				n_update, change = self.update(alpha[round], new_alpha[round], n_update, change)
				alpha[round] = new_alpha[round]
				n_update, change = self.update(beta[round], new_beta[round], n_update, change)
				beta[round] = new_beta[round]

			avg_change = change * 1.0 / n_update
			if avg_change < 0.1:
				break
		# print ('>>>>>>>>>>>E-step | iter:', counter)

		return q_z_i_0_, q_z_i_1_, alpha, beta, counter

	def m_step(self, q_z_i_0, q_z_i_1, features, classifier, alpha, beta):
		# prob_e_step = np.where(q_z_i_0 > 0.5, 0, 1)
		theta_i, classifier, counter = self.nn_em.train_m_step(classifier, features, q_z_i_1, self.conf['steps'], self.conf['m_epochs'])

		return theta_i, classifier, counter

	def aggragete_strategy(self, features, round_reputation, candidates_num):
		input_dim = features.shape[1]
		n_rounds = len(round_reputation)
		n_clients = int(self.conf['k'] + 2 * self.conf['num_pairs'])
		client_0 = client_1 = self.conf['num_pairs']
		y_train_0 = np.zeros((client_0, 1))
		y_train_1 = np.ones((client_1, 1))
		y_train = np.concatenate((y_train_0, y_train_1))

		round_reputation = np.array(round_reputation)
		sub_round_reputation = np.arange(n_rounds * n_clients).reshape(n_rounds, n_clients)
		for idx, num in enumerate(candidates_num):
			sub_round_reputation[:, idx] = round_reputation[:, num]

		q_z_i_0, q_z_i_1, A, B = self.init_probabilities(n_clients)
		alpha, beta = self.init_alpha_beta(A, B, n_rounds)
		if n_rounds > 1:
			classifier = tf.keras.models.load_model("./models_discriminator/classifier.h5")
		else:
			classifier = self.nn_em.define_nn(m=input_dim, n_neurons_1=self.conf['n_neurons_1'],
			                                  n_neurons_2=self.conf['n_neurons_2'], learning_rate=self.conf['m_lr'])

		# initialize the discriminator using the info of synthetic clients
		steps_it0 = 0
		epsilon = 1e-4
		theta_i = q_z_i_1.copy()
		old_theta_i = np.zeros((n_clients, 1))
		while (LA.norm(theta_i - old_theta_i) > epsilon) and (steps_it0 < self.conf['m_epochs']):
			classifier.fit(features[:int(client_0+client_1)], y_train, epochs=self.conf['steps'], verbose=0)
			theta_i_unlabeled = classifier.predict(features[len(y_train):])
			theta_i = np.concatenate((y_train, theta_i_unlabeled))
			steps_it0 += 1


		# perform variational EM
		em_step = 0
		epsilon = 1e-5
		e_step_counter_list = []
		m_step_counter_list = []
		old_theta_i = np.zeros((features.shape[0], 1))
		theta_i = np.ones((features.shape[0], 1))
		time_0 = time()
		# while em_step < self.conf['em_iterr'] and (LA.norm(theta_i - old_theta_i) > epsilon):
		while em_step < self.conf['em_iter']:
			# e-step
			q_z_i_0, q_z_i_1, alpha, beta, e_counter = self.e_step(n_rounds, n_clients, q_z_i_0, q_z_i_1, sub_round_reputation, \
														alpha, beta, theta_i, y_train=0, max_it=self.conf['e_iter'])
			e_step_counter_list.append(e_counter)
			old_theta_i = theta_i.copy()

			# m-step
			theta_i, classifier, m_counter = self.m_step(q_z_i_0, q_z_i_1, features, classifier, alpha, beta)
			m_step_counter_list.append(m_counter)

			em_step += 1

		return theta_i, sub_round_reputation, classifier, em_step, e_step_counter_list, m_step_counter_list, time()-time_0


	def model_aggregate(self, diff_record, total_num_clients):
		weight_accumulator = {}
		for name, params in self.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		total_samples = 0
		samples = []
		for i in range(len(diff_record)):
			total_samples += diff_record[i][1]
			samples.append(diff_record[i][1])
		prob_each_client = np.array(samples) / total_samples
		print ('>>>>prob_each_client:', prob_each_client)

		for num in range(total_num_clients):
			for name, params in self.global_model.state_dict().items():
				add_item = prob_each_client[num] * diff_record[num][0][name]

				if params.type() != add_item.type():
					weight_accumulator[name].add_(add_item.to(torch.int64))
				else:
					weight_accumulator[name].add_(add_item)

		for name, data in self.global_model.state_dict().items():
			data.add_(weight_accumulator[name])

	def model_change(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			update_per_layer = weight_accumulator[name]

			if data.type() != update_per_layer.type():
				data.copy_(update_per_layer.to(torch.int64))
			else:
				data.copy_(update_per_layer)

	def model_val(self):
		self.global_model.eval()

		total = 0
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		losses = []
		predictions = []
		Labels = []
		pre_prob = []
		for batch_id, batch in enumerate(self.val_loader):
			data, targets = batch
			dataset_size += data.size()[0]
			Labels.extend(targets.numpy())

			if torch.cuda.is_available():
				data = data.cuda()
				targets = targets.cuda()


			outputs = self.global_model(data)
			outputs_softmax = torch.softmax(outputs, 1)
			pre_prob.append(outputs_softmax.cpu().detach().numpy()[0])
			losses.extend(torch.nn.functional.cross_entropy(outputs, targets, reduction='none').cpu().detach().numpy())
			total_loss += torch.nn.functional.cross_entropy(outputs, targets,
											  reduction='sum').item() # sum up batch loss
			predicted = outputs.data.max(1)[1]  # get the index of the max log-probability
			predictions.extend(predicted.cpu().numpy())
			total += targets.size(0)
			correct += (predicted == targets).sum().item()

		acc = correct / total
		total_l = total_loss / total

		print(losses[:20])
		#print(np.array(pre_prob).reshape(-1, 10))
		print(Labels[:20])
		print(predictions[:20])

		return acc, total_l

	def model_test(self):
		self.global_model.eval()

		total = 0
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.test_loader):
			data, target = batch
			dataset_size += data.size()[0]

			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()


			output = self.global_model(data)

			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			total += target.size(0)
			correct += (pred == target).sum().item()

		acc = correct / total
		total_l = total_loss / total

		return acc, total_l
