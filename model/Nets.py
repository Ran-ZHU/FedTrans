
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self, dim_in, dim_hidden, dim_out):
		super(MLP, self).__init__()
		self.layer_input = nn.Linear(dim_in, dim_hidden)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout()
		self.layer_hidden = nn.Linear(dim_hidden, dim_out)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
		x = self.layer_input(x)
		x = self.dropout(x)
		x = self.relu(x)
		x = self.layer_hidden(x)
		return self.softmax(x)

class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()
		self.backbone = nn.Sequential(
		nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=2),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2, stride=2),
		nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2, stride=2),
		nn.Flatten(start_dim=1),
		nn.Linear(400, 120),
		nn.ReLU(),
		nn.Linear(120, 84),
		nn.ReLU(),
		nn.Linear(84, 10))
	def forward(self, x):
		x = self.backbone(x)
		return x

class CNNMnist(nn.Module):
	def __init__(self):
		super(CNNMnist, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes=10):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class CNNFashion_Mnist(nn.Module):
	def __init__(self):
		super(CNNFashion_Mnist, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc = nn.Linear(7*7*32, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out


class CNNCifar(nn.Module):
	def __init__(self):
		super(CNNCifar, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class modelC(nn.Module):
	def __init__(self, input_size, n_classes=10, **kwargs):
		super(modelC, self).__init__()
		self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
		self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
		self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
		self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
		self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
		self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
		self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
		self.conv8 = nn.Conv2d(192, 192, 1)

		self.class_conv = nn.Conv2d(192, n_classes, 1)


	def forward(self, x):
		x_drop = F.dropout(x, .2)
		conv1_out = F.relu(self.conv1(x_drop))
		conv2_out = F.relu(self.conv2(conv1_out))
		conv3_out = F.relu(self.conv3(conv2_out))
		conv3_out_drop = F.dropout(conv3_out, .5)
		conv4_out = F.relu(self.conv4(conv3_out_drop))
		conv5_out = F.relu(self.conv5(conv4_out))
		conv6_out = F.relu(self.conv6(conv5_out))
		conv6_out_drop = F.dropout(conv6_out, .5)
		conv7_out = F.relu(self.conv7(conv6_out_drop))
		conv8_out = F.relu(self.conv8(conv7_out))

		class_out = F.relu(self.class_conv(conv8_out))
		pool_out = F.adaptive_avg_pool2d(class_out, 1)
		pool_out.squeeze_(-1)
		pool_out.squeeze_(-1)
		return pool_out

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.conv_layer1 = self._make_conv_1(3,64)
        self.conv_layer2 = self._make_conv_1(64,128)
        self.conv_layer3 = self._make_conv_2(128,256)
        self.conv_layer4 = self._make_conv_2(256,512)
        self.conv_layer5 = self._make_conv_2(512,512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),    # 这里修改一下输入输出维度
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
            # 使用交叉熵损失函数，pytorch的nn.CrossEntropyLoss()中已经有过一次softmax处理，这里不用再写softmax
        )

    def _make_conv_1(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        return layer
    def _make_conv_2(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
              )
        return layer

    def forward(self, x):
        # 32*32 channel == 3
        x = self.conv_layer1(x)
        # 16*16 channel == 64
        x = self.conv_layer2(x)
        # 8*8 channel == 128
        x = self.conv_layer3(x)
        # 4*4 channel == 256
        x = self.conv_layer4(x)
        # 2*2 channel == 512
        x = self.conv_layer5(x)
        # 1*1 channel == 512
        x = x.view(x.size(0), -1)
        # 512
        x = self.classifier(x)
        # 10
        return x

n_channel=32

cfg = {
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, input_size=32, num_class=10):
        super(VGG, self).__init__()
        self.input_size = input_size
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier1 = nn.Linear(self.n_maps, 128)
        self.classifier2 = nn.Linear(128, num_class)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier1(out)
        out = self.classifier2(out)
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
                   nn.BatchNorm1d(self.n_maps),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


class SimpleCNN(nn.Module):
    def __init__(self, n_classes, input_size, fc_input, fc_output):
        super(SimpleCNN, self).__init__()
        self.n_classes = n_classes

        # CNN layer
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1)
        self.drop = nn.Dropout(p=0.6)
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)

        # Fully connected layer
        self.lin3 = nn.Linear(fc_input, fc_output)
        self.lin4 = nn.Linear(fc_output, self.n_classes)

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        if torch.cuda.is_available():
            x = x.cuda()

        batch_size = x.size(0)
        a = torch.relu(self.conv1(x))
        a = torch.relu(self.conv2(a))
        a = self.drop(a)
        a = self.pool(a)
        a = a.view((batch_size, -1))
        a = torch.relu(self.lin3(a))
        a = torch.relu(self.lin4(a))

        return a

class SimpleLSTM(nn.Module):
    def __init__(self, n_classes):
        super(SimpleLSTM, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=9,
            num_layers=2,
            batch_first=True
        )
        # Fully connected layer
        self.fc = nn.Linear(
            in_features=9,
            out_features=n_classes
        )

    def forward(self, x):
        x = x.type(torch.FloatTensor) #x: (batch_size, time_step, input_size) <- nn.LSTM(batch_first = True) False by default
        if torch.cuda.is_available():
            x = x.cuda()
        output_seq, _ = self.lstm(x) #output_seq: (batch_size, time_step, hidden_size) <- nn.LSTM(batch_first = True) False by default
        output_last = output_seq[:, -1, :] #output_last: (batch_size, hidden_size)
        output = self.fc(output_last)
        return output