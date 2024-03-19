# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from .Nets import *
from torchvision import models


def get_model(name="vgg16", num_classes=10, input_size=9, fc_input=1984, fc_output=100, pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
		dim = model.fc.in_features
		model.fc = torch.nn.Linear(dim, num_classes)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
		dim = model.fc.in_features
		model.fc = torch.nn.Linear(dim, num_classes)
	elif name == "mobilenet_v2":
		model = models.mobilenet_v2(pretrained=pretrained)
		model.classifier[0] = nn.Linear(model.classifier[1].in_features, 128)
		model.classifier[1] = nn.Linear(128, num_classes)
		# num_ftrs = model.classifier[-1].in_features
		# model.classifier[-1] = torch.nn.Linear(num_ftrs, 10)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
		dim = model.fc.in_features
		model.fc = torch.nn.Linear(dim, num_classes)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
		dim = model.fc.in_features
		model.fc = torch.nn.Linear(dim, num_classes)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
		dim = model.fc.in_features
		model.fc = torch.nn.Linear(dim, num_classes)
	elif name == "cnn_cifar":
		model = CNNCifar()
	elif name == "cnn_mnist":
		model = CNNMnist()
	elif name == "lenet":
		model = LeNet5()
	elif name == "cnn_femnist":
		model = FemnistCNN(num_classes=num_classes)
	elif name == "simple_cnn":
		model = SimpleCNN(n_classes=num_classes, input_size=input_size, fc_input=fc_input, fc_output=fc_output)
	elif name == "simple_lstm":
		model = SimpleLSTM(n_classes=num_classes)
		
		
	if torch.cuda.is_available():
		# summary(model, (3, 32, 32))
		return model.cuda()
	else:
		# summary(model, (3, 32, 32))
		return model







