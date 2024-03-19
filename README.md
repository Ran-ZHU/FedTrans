# <h1 align="center"> FedTrans: Client-Transparent Utility Estimation for Robust Federated Learning</h1>


[//]: # (<p align="center">)

[//]: # (<a href="https://arxiv.org/abs/2310.05078"><img src="https://img.shields.io/badge/arxiv-2310.05078-silver" alt="Paper"></a>)

[//]: # (<a href="https://iclr.cc/"><img src="https://img.shields.io/badge/Pub-ICLR'24-olive" alt="Pub"></a>)

[//]: # (<a href="https://github.com/Ran-ZHU/FedTrans"><img src="https://img.shields.io/badge/-github-teal?logo=github" alt="github"></a>)

[//]: # (</p>)

![Generic badge](https://img.shields.io/badge/code-official-green.svg)  This repository is the official implementation of "[ShuffleFL: Addressing Heterogeneity in Multi-Device Federated Learning](https://openreview.net/pdf?id=DRu8PMHgCh)"
in the International Conference on Learning Representations (ICLR) 2024.

## Introduction
Federated Learning (FL) is an important privacy-preserving learning paradigm that plays an important role in the Intelligent Internet of Things. 
Training a global model in FL, however, is vulnerable to the data noise across the clients. In this paper, we introduce **FedTrans**, a novel 
client-transparent client utility estimation method designed to guide client selection for noisy scenarios, mitigating performance degradation 
problems. To estimate the client utility, we propose a Bayesian framework that models client utility and its relationships with the weight 
parameters and the performance of local models. We then introduce a variational inference algorithm to effectively infer client utility at the 
FL server, given only a small amount of auxiliary data. Our evaluation results demonstrate that leveraging FedTrans to select the clients can 
improve the accuracy performance (up to 7.8\%), ensuring the robustness of FL in noisy scenarios.

![overview](./overview.png)

## Contents
This repo includes the implementation of FedTrans under three different local data distributions (H2C/Dir/IID) on CIFAR10 and Fashion-MNIST datasets.

## Requirements

To install requirements：

```
pip install -r requirements.txt
```


## Repo structure

## How to run

To train the model with non-IID dataset in this paper, run the command:
```
python main.py
```

## Parameters
In [conf.json](./conf.json), you can change the hyper-parameters and some settings. Here we give the detailed 
description for each parameter defined in [conf.json](./utils/conf.json):

| Parameter                      | Description                                 |
| -------------------------- | ---------------------------------- |
| `if_shuffling`|     whether using ShuffleFL|
| ` load_client_dict`|     whether loading the stored client dict|
| ` type`|      dataset to use |
| ` model_name`|      model architecture|
| `comm_mobility`|     whether using dynamic communication capacity|
| ` comm_reduce_rate`|     the coefficient of varying communication capacities|
| ` no_users`|     number of users|
| ` no_models`|   number of devices|
| ` balance_weight`|      the coefficient [\alpha] to scale the data imbalance term|
| ` selection_rate`|      the ratio of users selected in each communication round|
| ` concentration_device`|     the concentration parameter of the Dirichlet distribution for non-IID partition among devices|
| ` global_epochs`|     number of global epochs|
| `local_epochs`|     number of local repochs|
| ` lr`|      learning rate|
| ` seed`|     The seed to generate the user candiadate involved in each round|


## Citation
If our work is userful for your research, please consider citing:
```

```
