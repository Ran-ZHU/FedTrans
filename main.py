# -*- coding: utf-8 -*-

import pandas as pd
import argparse, json
import datetime
import pickle
import os
import math
import logging
import numpy as np
from time import time
from torchsummary import summary
from sklearn import preprocessing
from torchvision import transforms, datasets
from server import *
from client import *
from utils.setup_utils import *
from synthetic_clients import *
from copy import deepcopy
from utils.sampling import iid_partition, auxi_data_for_synthetic_clients, h2c_partition, dirichlet_distribution_fashion


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='FedTrans')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)

	conf['num_classes'], conf['model_name'], conf['num_models'], conf['metric'], conf['record_file_name'],\
		conf['client_info_file'], conf['em_info_file'] = get_other_conf(conf)

	# load dataset and split users
	if conf['data'] == 'fashionmnist':
		train_dataset = datasets.FashionMNIST('./data/fashionmnist/', train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.FashionMNIST('./data/fashionmnist/', train=False, download=True, transform=transforms.ToTensor())
		# used for constructing open-set noise type by introducing data from other source while keeping the labels unchanged
		data_for_openset_noise = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transforms.ToTensor())
		top_layer_feature_name = "output.weight"
		# server = Server(conf, eval_datasets, val_indices)
		# summary(server.global_model, (1, 28, 28))
	elif conf['data'] == 'cifar10':
		train_dataset = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transforms.ToTensor())
		# used for constructing open-set noise type by introducing data from other source while keeping the labels unchanged
		data_for_openset_noise = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transforms.ToTensor())
		top_layer_feature_name = "classifier.1.weight"
		# server = Server(conf, eval_datasets, val_indices)
		# summary(server.global_model, (3, 32, 32))
	else:
		exit('Error: unrecognized dataset')

	# construct auxiliary data sampled from evaluation dataset
	num_synthetic_clients = int(2 * conf['num_pairs'])
	dict_synthetic_clients, test_indices, auxiliary_data_indices = \
		auxi_data_for_synthetic_clients(conf, eval_dataset, num_synthetic_clients)

	if conf['data_distribution'] == "iid":
		dict_clients = iid_partition(train_dataset, conf['num_models'])
	elif conf['data_distribution'] == "dirichlet":
		dict_clients = dirichlet_distribution_fashion(train_dataset, conf['num_classes'], conf['num_models'], 0.5)
	elif conf['data_distribution'] == "h2c":
		dict_clients = h2c_partition(train_dataset, conf['num_models'], conf['num_classes'])
	dict_clients_for_openset_noise = dict_clients


	random.seed(222)
	np.random.seed(333)
	noisy_rate_list = noise_rate(conf)  # load noise rate for noisy clients
	candidate_table = involved_client_table(conf)  # load candidates involved in each round

	# initialize the server
	server = Server(conf, eval_dataset, auxiliary_data_indices, test_indices)

	# initialize clients
	clients = []
	for client_id in range(conf["num_models"]):
		clients.append(Client(conf=conf, train_dataset=train_dataset, data_for_openset_noise=data_for_openset_noise, eval_dataset=eval_dataset,
		                      dict_clients=dict_clients, val_indices=auxiliary_data_indices, dict_clients_for_openset_noise=dict_clients_for_openset_noise,
		                      noisy_ratio_list=noisy_rate_list, client_id=client_id))
	print("\n\n")

	# initialize synthetic clients
	synthetic_clients = []
	for c_l_0 in range(conf['num_pairs']):
		synthetic_clients.append(Synthetic_clients(conf, eval_dataset, dict_synthetic_clients, auxiliary_data_indices, c_l_0, label='0')) # label 0 means noisy clients
	for c_l_1 in range(conf['num_pairs'], num_synthetic_clients):
		synthetic_clients.append(Synthetic_clients(conf, eval_dataset, dict_synthetic_clients, auxiliary_data_indices, c_l_1, label='1')) # label 1 means clean clients


	# FL training
	epoch_index = -1
	record_global_model = []
	theta_record = []
	round_reputation = []
	record_selectedNum = []
	all_acc_list = []
	all_loss_list = []
	em_info_list = []
	client_info_list = []  # record the client info during training
	for e in range(conf['global_epochs']):
		tmp_data = []
		torch.cuda.empty_cache()
		print ('*****************ROUND %d*****************' %(e+1))
		time_begin = time()
		candidates_current_round = candidate_table[e]

		# load the candidate clients involved in current round
		candidates_list = []
		candidates_num = []
		for i in range(len(synthetic_clients)):
			candidates_list.append(synthetic_clients[i])
			candidates_num.append(conf['num_models']+i)
		for i in range(len(candidates_current_round)):
			candidates_list.append(clients[candidates_current_round[i]])
			candidates_num.append(candidates_current_round[i])
		print('======Client ID involved in current round======= \n', candidates_num)

		acc_record = []
		loss_record = []
		All_features = []
		diff_record = {}
		em_info_dict = {}
		client_info_dict = {c: {} for c in candidates_num}  # record the candidate info of current round
		for idx, c in enumerate(candidates_list):
			# print the intermediate info
			if candidates_num[idx] < conf['num_models']:
				print('>>>Client %d begin training>>>' % candidates_num[idx])
			elif (candidates_num[idx]-conf['num_models']) < conf['num_pairs']:
				print('>>>Synthetic noisy client %d begin training>>>' % candidates_num[idx])
			else:
				print('>>>Synthetic clean client %d begin training>>>' % candidates_num[idx])

			# store the client info
			client_id = candidates_num[idx]
			client_info_dict[client_id]["client_id"] = client_id
			if client_id >= int(conf["num_models"] * conf["noisy_client_rate"]):
				client_info_dict[client_id]["noise_rate"] = 0
			else:
				client_info_dict[client_id]["noise_rate"] = noisy_rate_list[client_id]

			# perform local training
			diff, local_data_length, val_acc, val_loss, client_label = c.local_train(server.global_model)
			print(">>>>>>>Client label:", client_label) # 0 means noisy client, 1 means clean client, 'synthetic' means synthetic client
			acc_record.append(val_acc)
			loss_record.append(val_loss)
			c.local_model.zero_grad()
			diff_record[idx] = (diff, local_data_length)
			tmp_data.append(client_label)
			print ("Round: %d, Val_acc: %f, Val_loss: %f \n" % (e+1, val_acc, val_loss))

			features = []
			for name, params in server.global_model.state_dict().items():
				if name == top_layer_feature_name:
					features = diff[name].cpu().detach().numpy().reshape(1, -1)
			All_features.append(features)

		# construct round-reputation matrix
		acc_record = np.array(acc_record)
		loss_record = np.array([np.min(loss_record) if np.isnan(l) else l for l in loss_record])
		## using accuracy/loss as metric
		single_round = [2] * int(conf["num_models"] + num_synthetic_clients) # 2 here means the corresponding client in current round is not selected
		if conf["metric"] == "acc":
			print (">>>>>>>>>>>>>>Using Acc as metric")
			diff_acc = []
			tmp_mean = np.mean(acc_record[(num_synthetic_clients):])
			for i in range(conf['k']):
				tmp = acc_record[i+num_synthetic_clients] - tmp_mean
				diff_acc.append(tmp)
				if tmp >= 0:
					single_round[candidates_num[i + num_synthetic_clients]] = 1  # 1 means corresponding clients are clean
				else:
					single_round[candidates_num[i + num_synthetic_clients]] = 0  # 1 means corresponding clients are noisy

			for i in range(conf['num_pairs']):
				single_round[conf['num_models'] + i] = 0
			for j in range(conf['num_pairs']):
				single_round[conf['num_models'] + conf['num_pairs'] + j] = 1
		elif conf["metric"] == "loss":
			print(">>>>>>>>>>>>>>Using Loss as metric")
			# using loss as metrics
			diff_loss = []
			tmp_mean = np.mean(loss_record[(num_synthetic_clients):])
			for i in range(conf['k']):
				tmp = loss_record[i+num_synthetic_clients] - tmp_mean
				diff_loss.append(tmp)
				if tmp >= 0:
					single_round[candidates_num[i+num_synthetic_clients]] = 1
				else:
					single_round[candidates_num[i+num_synthetic_clients]] = 0

			for i in range(conf['num_pairs']):
				single_round[conf['num_models'] + i] = 0
			for j in range(conf['num_pairs']):
				single_round[conf['num_models'] + conf['num_pairs'] + j] = 1
		round_reputation.append(single_round)

		# obtain the input of weight-based discriminator
		All_features = np.array(All_features).reshape(len(candidates_num), -1)
		All_features = preprocessing.scale(All_features)


		# FedTrans
		theta_i, sub_matrix, classifier, em_counter, e_step_counter_list, m_step_counter_list, em_time = \
			server.aggragete_strategy(All_features, round_reputation, candidates_num)
		# save the discriminator for the next round
		classifier.save("./models_discriminator/classifier.h5")
		# record the info of em algorithm
		em_info_dict["em_iter_num"] = em_counter
		em_info_dict["em_time"] = em_time
		em_info_dict["e_step_counter_list"] = e_step_counter_list
		em_info_dict["m_step_counter_list"] = m_step_counter_list
		em_info_list.append(em_info_dict)
		print ("matrix:\n", sub_matrix)
		print ('Final theta\n', theta_i)
		theta_record.append(theta_i[num_synthetic_clients:])

		for c_idx, utility in enumerate(theta_i[num_synthetic_clients:]):
			client_info_dict[candidates_num[c_idx]]["utility"] = utility
			if utility > 0.5:
				client_info_dict[candidates_num[c_idx]]["result_0.5"] = 1
			else:
				client_info_dict[candidates_num[c_idx]]["result_0.5"] = 0

		client_info_list.append(client_info_dict)


		# perform client selection
		final_selection = []
		selected_idx = []
		for t_idx, theta in enumerate(theta_i[num_synthetic_clients:]):
			if theta >= conf['theta_threshold']:
				final_selection.append(1)
				selected_idx.append(num_synthetic_clients+t_idx)
			else:
				final_selection.append(0)

		final_selection = np.array(final_selection)
		tmp_data = np.array(tmp_data[num_synthetic_clients:])
		print ("True label", tmp_data)
		print ("Pred label", final_selection)
		print (candidates_num)


		# update the round-reputation matrix
		update_round = np.arange(conf['num_models']+num_synthetic_clients)
		update_round[:] = 2
		for i in range(len(candidates_num)-num_synthetic_clients):
			if final_selection[i] == 1:
				update_round[candidates_num[i+num_synthetic_clients]] = 1
			else:
				update_round[candidates_num[i+num_synthetic_clients]] = 0

		for c_0 in range(conf['num_pairs']):
			update_round[conf['num_models']+c_0] = 0
		for c_1 in range(conf['num_pairs']):
			update_round[conf['num_models']+conf['num_pairs']+c_1] = 1

		round_reputation[-1] = update_round

		# perform model aggregation
		if len(selected_idx) != 0:
			diff_record_final = {}
			n = 0
			for c_idx in selected_idx:
				diff_record_final[n] = diff_record[c_idx]
				n += 1
			diff_record = {}
			server.model_aggregate(diff_record_final, total_num_clients=len(selected_idx))
			Val_acc, Val_loss = server.model_val()
			Test_acc, Test_loss = server.model_test()
			print ("Round %d, Val_acc: %f, Val_loss: %f\n" % (e+1, Val_acc, Val_loss))
			print ("Round %d, Test_acc: %f, Test_loss: %f\n" % (e+1, Test_acc, Test_loss))
			all_acc_list.append(Test_acc)
			all_loss_list.append(Test_loss)


			record_global_model.append([e+1, Val_acc, Val_loss, Test_acc, Test_loss, conf['data'], conf['data_distribution'], conf['model_name'],
			                            conf['num_models'], conf['k'], conf['noisy_client_rate'], conf['noise_rate'], conf['noise_type'],
			                            conf['auxiliary_data_len'], conf['metric'], conf['theta_threshold'], conf['num_pairs'],
			                            conf['lr'], conf['batch_size'], conf['local_epochs']])
			print('>>>>>>>Time Cost:', time() - time_begin)
		else:
			print ("Round %d is invalid\n" %(e + 1))
			if e != 0:
				round_reputation = round_reputation[:-1]
				buffer_latest_record = deepcopy(record_global_model[-1])
				record_global_model.append(buffer_latest_record)
				record_global_model[-1][0] = e + 1

		if (e+1) % 1 == 0:
			pd.DataFrame(record_global_model).to_csv(conf['record_file_name'], index=False,
													header=['Round', 'Val_acc', 'Val_loss', 'Test_acc', 'Test_loss', 'Dataset', 'Data_distribution',
													        'Model_name', 'Total_clients', 'Num_clients', 'Noisy_client_rate', 'Noise rate', 'Noise_type',
													        'Auxiliary_data_len', 'Metric', 'Threshold', 'Num_pairs', 'Lr', 'Batch_size', 'Local_epochs'])


			with open(conf['client_info_file'], 'wb') as f:
				pickle.dump(client_info_list, f)

			with open(conf['em_info_file'], 'wb') as f:
				pickle.dump(em_info_list, f)


