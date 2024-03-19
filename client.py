# -*- coding: utf-8 -*-

from torch import nn
import torch, copy
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, sampler
import random

import numpy as np
import math
from utils.insert_noise import *
from model.models import get_model


class Client(object):
    def __init__(self, conf, train_dataset, data_for_openset_noise, eval_dataset, dict_clients, val_indices,
                 dict_clients_for_openset_noise, noisy_ratio_list, client_id=-1):
        self.conf = conf
        self.local_model = get_model(self.conf["model_name"], num_classes=self.conf['num_classes'])
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.data_for_openset_noise = data_for_openset_noise
        self.eval_dataset = eval_dataset
        self.dict_clients = dict_clients
        self.dict_clients_for_open_noise = dict_clients_for_openset_noise
        self.noisy_ratio_list = noisy_ratio_list
        self.val_indices = val_indices

        # assign noise rate to client
        if self.conf['noise_rate'] == "random":
            if self.client_id < int(self.conf['num_models'] * self.conf['noisy_client_rate']):
                self.noise_rate = self.noisy_ratio_list[self.client_id]
                print('The noise rate of Client %d: %f' % (self.client_id, self.noise_rate))
            else:
                self.noise_rate = 0
        else:
            if self.client_id < int(self.conf['num_models'] * self.conf['noisy_client_rate']):
                self.noise_rate = self.conf['noise_rate']
                print('The noise rate of Client %d: %f' % (self.client_id, self.noise_rate))
            else:
                self.noise_rate = 0

        #  randomly insert noise into local data
        self.noisy_samples_id = np.random.choice(len(self.dict_clients[self.client_id]),
                                                int(len(self.dict_clients[self.client_id]) * self.noise_rate),
                                                replace=False)

        if self.conf['noise_type'] == "hybrid_across":
            self.train_loader = Dataset_noise_hybrid_across(self.conf, self.train_dataset, self.dict_clients, self.data_for_openset_noise,
                                        self.dict_clients_for_open_noise, self.noisy_samples_id, self.client_id)
        elif self.conf['noise_type'] == "hybrid_intra":
            self.train_loader = DataLoader(Dataset_noise_hybrid_intra(self.conf, self.train_dataset, self.dict_clients[self.client_id],
                                                                      self.data_for_openset_noise, self.dict_clients_for_open_noise[self.client_id],
                                                                      self.noisy_samples_id),
                                           batch_size=conf["batch_size"], shuffle=True, drop_last=True)
        elif self.conf['noise_type'] == "label_across":
            self.train_loader = Dataset_noise_label_across(self.conf, self.train_dataset, self.dict_clients, self.data_for_openset_noise,
                                                           self.dict_clients_for_open_noise, self.noisy_samples_id, self.client_id)
        elif self.conf['noise_type'] == "label_intra":
            self.train_loader = DataLoader(Dataset_noise_label_intra(self.conf, self.train_dataset, self.dict_clients[self.client_id],
                                                                      self.data_for_openset_noise, self.dict_clients_for_open_noise[self.client_id],
                                                                      self.noisy_samples_id),
                                           batch_size=conf["batch_size"], shuffle=True, drop_last=True)
        elif self.conf['noise_type'] == "feature_across":
            self.train_loader = Dataset_noise_feature_across(self.conf, self.train_dataset, self.dict_clients,
                                                             self.noisy_samples_id, self.client_id)
        elif self.conf['noise_type'] == "feature_intra":
            self.train_loader = DataLoader(Dataset_noise_feature_intra(self.conf, self.train_dataset,
                                                                       self.dict_clients[self.client_id], self.noisy_samples_id),
                                           batch_size=conf["batch_size"], shuffle=True, drop_last=True)
        else:
            # only random flipping noise
            print ("Only inserting random flipping noise!")
            self.train_loader = DataLoader(
                DatasetSplit_flip(self.conf, self.train_dataset, self.dict_clients[self.client_id],
                                  samples_flip_id=self.noisy_samples_id), batch_size=self.conf['batch_size'], shuffle=True, drop_last=True)


        self.val_loader = DataLoader(self.eval_dataset, batch_size=self.conf["batch_size"], \
                                                           sampler=sampler.SubsetRandomSampler(self.val_indices))


    def local_train(self, model):
        local_data_length = len(self.dict_clients[self.client_id])
        original_labels = np.array(self.train_dataset.targets)[list(self.dict_clients[self.client_id])].tolist()
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

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])


        # validate local model
        correct, total, loss = 0, 0, 0.0
        predictions = []
        Labels = []
        pre_prob = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.val_loader):
                data, targets = batch
                Labels.extend(targets.numpy())

                if torch.cuda.is_available():
                    data = data.cuda()
                    targets = targets.cuda()

                outputs = self.local_model(data)
                outputs_softmax = torch.softmax(outputs, 1)
                pre_prob.append(outputs_softmax.cpu().numpy()[0])

                loss += torch.nn.functional.cross_entropy(outputs, targets, reduction='sum').item()
                predicted = outputs.data.max(1)[1]
                predictions.extend(predicted.cpu().numpy())
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_acc = correct / total
        val_loss = loss / total


        if self.client_id < int(self.conf['num_models']*self.conf['noisy_client_rate']):
            client_label = 0
        else:
            client_label = 1

        return diff, local_data_length, val_acc, val_loss, client_label
