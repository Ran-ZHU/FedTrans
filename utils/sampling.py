# -*- coding: utf-8 -*-
import os
import numpy as np
from torchvision import datasets, transforms
import pandas as pd
from generate_csv import generate_raw_data

def iid_partition(dataset, num_clients):
    """
    Sample I.I.D. client data from FashionMnist/Cifar10 dataset
    :param dataset: training data
    :param num_clients: the number of clients participating in FL
    :return: dict of image index
    """
    length = int(len(dataset))
    num_items = int(length/num_clients)
    dict_users, all_idxs = {}, [i for i in range(length)]
    for i in range(num_clients):
        dict_users[i] = np.random.choice(all_idxs, num_items,
                                             replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

def auxi_data_for_synthetic_clients(conf, eval_dataset, num_synthetic_clients):
    """
       Sample data from evaluation data as auxiliary data
       to construct synthetic clients, the remaining as test data
    :param conf:
    :param eval_dataset: evaluation data
    :param num_synthetic_clients: the number of synthetic clients
    :return: dict of image index
    """
    auxiliary_data_len = conf['auxiliary_data_len']
    num_classes = conf["num_classes"]
    num_per_class = int(auxiliary_data_len / num_classes)
    num_per_class_per_client = int(num_per_class / num_synthetic_clients)
    dict_synthetic = {i:[] for i in range(num_synthetic_clients)}

    auxiliary_data_indices = []
    for c in range(num_classes):
        auxiliary_data_indices_tmp = []
        for idx, data in enumerate(eval_dataset):
            if len(auxiliary_data_indices_tmp) >= num_per_class:
                break
            if data[1] == c:
                auxiliary_data_indices_tmp.append(idx)
        for i in range(num_synthetic_clients):
            dict_synthetic[i].extend(auxiliary_data_indices_tmp[i*num_per_class_per_client: (i+1)*num_per_class_per_client])

        auxiliary_data_indices.extend(auxiliary_data_indices_tmp)
    test_indices = list(set([i for i in range(len(eval_dataset))])-set(auxiliary_data_indices))


    return dict_synthetic, test_indices, auxiliary_data_indices

def dirichlet_distribution_fashion(dataset, num_class, num_client, concentration):
    np.random.seed(333)
    x, y = dataset.data, np.array(dataset.targets)
    print (x[0].shape)
    n_sample = y.shape[0]
    n_max_sample = int(n_sample / num_client)
    idx_sample_to_client = [[] for _ in range(num_client)]
    for idx_class in range(num_class):
        sample_in_idx_class = np.where(y == idx_class)[0]
        np.random.shuffle(sample_in_idx_class)
        initial_dirichlet = np.random.dirichlet(np.repeat(concentration, num_client))
        dirichlet = np.array([p * (len(idx_client) < n_max_sample) for p, idx_client in zip(initial_dirichlet, idx_sample_to_client)])
        weight = dirichlet / np.sum(dirichlet)
        n_sample_per_client = (np.cumsum(weight) * len(sample_in_idx_class)).astype(int)[:-1]
        idx_sample_to_client = [idx_client + idx_sample.tolist() for idx_client, idx_sample in zip(idx_sample_to_client, np.split(sample_in_idx_class, n_sample_per_client))]


    dict_clients = {i: np.array(idx_sample) for i, idx_sample in zip(np.arange(num_client), idx_sample_to_client)}

    for i in range(num_client):
        print (len(dict_clients[i]))

    return dict_clients

def h2c_partition(dataset, num_clients, num_class):
    np.random.seed(333)
    num_shards = int(num_clients * 2)
    num_imgs = int(len(dataset) / (num_shards))
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_clients):
        rand_set = []
        shard_per_class = num_shards/num_class

        rand_1 = np.random.choice(idx_shard, 1)
        rand_set.extend(rand_1)
        idx_shard = list(set(idx_shard) - set(rand_1))

        rand_2 = np.random.choice(idx_shard, 1)
        while int(rand_1 / shard_per_class) == int(rand_2 / shard_per_class):
            rand_2 = np.random.choice(idx_shard, 1)
        rand_set.extend(rand_2)

        idx_shard = list(set(idx_shard) - set(rand_2))

        for rand in rand_set:
            dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0).astype(np.int)
    return dict_clients


