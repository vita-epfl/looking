import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms

torch.manual_seed(1)
np.random.seed(0)

class Binarize(nn.Module):
    def __init__(self):
        super(Binarize, self).__init__()
    def forward(self, x):
        return torch.where(x > 0, 1, 0).float()

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class LookingModel(nn.Module):
    def __init__(self, input_size, p_dropout=0.2, output_size=1, linear_size=256, num_stage=3, bce=False):
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
        #y = self.binarize(y)
        y = self.w2(y)
        #y = self.binarize(y)
        y = self.sigmoid(y)
        return y


class Linear(nn.Module):
    def __init__(self, linear_size=256, p_dropout=0.2):
        super(Linear, self).__init__()
        ###

        self.linear_size = linear_size
        self.p_dropout = p_dropout

        ###

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(self.p_dropout)
        #self.binarize = Binarize()

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

class AlexNet_heads(nn.Module):
    def __init__(self, device, fine_tune=False):
        super(AlexNet_heads, self).__init__()
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

class ResNet18_heads(nn.Module):
    def __init__(self, device):
        super(ResNet18_heads, self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.net.fc  = nn.Sequential(
            nn.Linear(in_features=512, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.net(x)

class ResNet50_heads(nn.Module):
    def __init__(self, device):
        super(ResNet50_heads, self).__init__()
        net = models.resnext50_32x4d(pretrained=True)
        net.fc  = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.net = net

    def forward(self, x):
        return self.net(x)


class LookingNet_early_fusion_eyes(nn.Module):
    def __init__(self, PATH, PATH_look, input_size, device, fine_tune=True):
        super(LookingNet_early_fusion_eyes, self).__init__()
        self.eyes = LookingModel(450)
        if fine_tune:
            self.eyes.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            for m in self.eyes.parameters():
                m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(input_size)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()


        self.final = nn.Sequential(
            nn.Linear(512, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        eyes, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook

        # End of third linear stage
        self.eyes.dropout.register_forward_hook(get_activation('eyes'))
        self.looking_model.dropout.register_forward_hook(get_activation('look'))

        output_eyes = self.eyes(eyes)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_eyes = activation["eyes"]

        out_final = torch.cat(layer_eyes, layer_look, 1).type(torch.float)

        return self.final(out_final)

class LookingNet_late_fusion_eyes(nn.Module):
    def __init__(self, PATH, PATH_look, input_size, device, fine_tune=True):
        super(LookingNet_late_fusion_eyes, self).__init__()
        self.eyes = LookingModel(450)
        if fine_tune:
            self.eyes.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            for m in self.eyes.parameters():
                m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(input_size)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()


        self.final = nn.Sequential(
            nn.Linear(512, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        eyes, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook

        # End of third linear stage
        self.eyes.linear_stages[2].bn2.register_forward_hook(get_activation('eyes'))
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))

        output_eyes = self.eyes(eyes)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_eyes = activation["eyes"]

        out_final = torch.cat(layer_eyes, layer_look, 1).type(torch.float)

        return self.final(out_final)


class LookingNet_late_fusion_18(nn.Module):
    def __init__(self, PATH, PATH_look, input_size, device, fine_tune=True):
        super(LookingNet_late_fusion_18, self).__init__()
        self.backbone = ResNet18_heads(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH))
            for m in self.backbone.net.parameters():
                m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(input_size)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Linear(272, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        head, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook
        def get_activation2(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))

        #x = torch.randn(1, 25)
        output_head = self.backbone(head)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), layer_look), 1).type(torch.float)

        return self.final(out_final)

class LookingNet_early_fusion_18(nn.Module):
    def __init__(self, PATH, PATH_look, input_size, device, fine_tune=True):
        super(LookingNet_early_fusion_18, self).__init__()
        self.backbone = ResNet18_heads(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH))
            for m in self.backbone.net.parameters():
                m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(input_size)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Linear(272, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        head, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook

        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        self.looking_model.dropout.register_forward_hook(get_activation('look'))

        #x = torch.randn(1, 25)
        output_head = self.backbone(head)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), layer_look), 1).type(torch.float)

        return self.final(out_final)


class LookingNet_late_fusion_50(nn.Module):
    def __init__(self, PATH, PATH_look, input_size, device, fine_tune=True):
        super(LookingNet_late_fusion_50, self).__init__()
        self.backbone = ResNet50_heads(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH))
            for m in self.backbone.parameters():
                m.requires_grad = False
            self.backbone = self.backbone.eval()

        self.looking_model = LookingModel(input_size)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model = self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(2048, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Linear(272, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        head, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook

        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))

        #x = torch.randn(1, 25)
        output_head = self.backbone(head)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), layer_look), 1).type(torch.float)

        return self.final(out_final)

class LookingNet_early_fusion_50(nn.Module):
    def __init__(self, PATH, PATH_look, input_size, device, fine_tune=True):
        super(LookingNet_early_fusion_50, self).__init__()
        self.backbone = ResNet50_heads(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH))
            for m in self.backbone.parameters():
                m.requires_grad = False
            self.backbone = self.backbone.eval()

        self.looking_model = LookingModel(input_size)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model = self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(2048, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Linear(272, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        head, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook

        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        self.looking_model.dropout.register_forward_hook(get_activation('look'))

        #x = torch.randn(1, 25)
        output_head = self.backbone(head)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), layer_look), 1).type(torch.float)

        return self.final(out_final)
