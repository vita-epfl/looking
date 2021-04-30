import torch.nn as nn
import torch
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class LookingModel(nn.Module):
    def __init__(self, input_size, output_size=1, linear_size=256, p_dropout=0.2, num_stage=3, bce=False):
        super(LookingModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.bce = bce
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        y = self.sigmoid(y)
        return y


class Linear(nn.Module):
    def __init__(self, linear_size=256, p_dropout=0.2):
        super(Linear, self).__init__()

        ###

        self.linear_size = linear_size
        self.p_dropout = p_dropout

        ###

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        ###
        self.l1 = nn.Linear(self.linear_size, self.linear_size)
        self.bn1 = nn.BatchNorm1d(self.linear_size)

        self.l2 = nn.Linear(self.linear_size, self.linear_size)
        self.bn2 = nn.BatchNorm1d(self.linear_size)
        
    def forward(self, x):
        # stage I

        y = self.l1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # stage II

        
        y = self.l2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.dropout(y)

        # concatenation

        out = x+y

        return out

class AlexNet_head(nn.Module):
    def __init__(self, device, fine_tune=False):
        super(AlexNet_head, self).__init__()
        net = models.alexnet(pretrained=True).to(device)
        net.classifier  = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1),
                    nn.Sigmoid()
        ).to(device)
        if fine_tune:
            for param in net.parameters():
                param.requires_grad = False
            for param in net.classifier.parameters():
                param.requires_grad = True

        self.net = net
    def forward(self, x):
        return self.net(x)

class ResNet18_head(nn.Module):
    def __init__(self, device):
        super(ResNet18_head, self).__init__()
        net = models.resnet18(pretrained=True)
        net.fc  = nn.Sequential(
            nn.Linear(in_features=512, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.net(x)

class ResNet50_head(nn.Module):
    def __init__(self, device):
        super(ResNet50_head, self).__init__()
        net = models.resnext50_32x4d(pretrained=True)
        net.fc  = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.net = net
    
    def forward(self, x):
        return self.net(x)