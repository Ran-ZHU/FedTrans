# -*- coding: utf-8 -*-

import numpy as np



def get_other_conf(conf):
    if conf['data'] == "fashionmnist":
        num_classes = 10
        model_name = "cnn_femnist"
        num_models = 100
    elif conf['data'] == "cifar10":
        num_classes = 10
        model_name = "mobilenet_v2"
        num_models = 100

    if conf['data_distribution'] == "h2c":
        metric = "loss"
    else:
        metric = "acc"

    change_term = conf['record_file_name'].split('/')[-1]
    record_file_name = conf['record_file_name'].replace(change_term, "{}_{}_{}_{}".format(conf['data'], conf['noise_type'],
                                                                                          conf['data_distribution'], change_term))
    client_info_file_name = \
        conf['record_file_name'].replace(change_term,
                                         "{}_{}_{}_client_info_{}".
                                         format(conf['data'], conf['noise_type'],conf['data_distribution'],
                                                change_term)).replace("csv", "data")
    em_info_file_name = \
        conf['record_file_name'].replace(change_term,
                                         "{}_{}_{}_em_info_{}".
                                         format(conf['data'], conf['noise_type'], conf['data_distribution'],
                                                change_term)).replace("csv", "data")

    return num_classes, model_name, num_models, metric, record_file_name, client_info_file_name, em_info_file_name

def noise_rate(conf):
    np.random.seed(333)
    noise_rate_list = np.random.choice(np.arange(0.1, 1.05, 0.1),
                                       int(conf['num_models'] * conf['noisy_client_rate']), replace=True)
    return noise_rate_list

def involved_client_table(conf):
    np.random.seed(345)
    table = []
    for i in range(conf['global_epochs']):
        table.append(np.random.choice(np.arange(conf['num_models']), size=conf['k'], replace=False))

    return table
