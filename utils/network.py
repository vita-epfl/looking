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
    """
    Class definition for the Looking Model.
    """
    def __init__(self, input_size, p_dropout=0.2, output_size=1, linear_size=256, num_stage=3):
        """[summary]

        Args:
            input_size (int): Input size for the model. If the whole pose needs to be used, the value should be 51.
            p_dropout (float, optional): Dropout rate in the linear blocks. Defaults to 0.2.
            output_size (int, optional): Output number of nodes. Defaults to 1.
            linear_size (int, optional): Size of the fully connected layers in the Linear blocks. Defaults to 256.
            num_stage (int, optional): Number of stages to use in the Linear Block. Defaults to 3.
            bce (bool, optional): Make use of the BCE Loss. Defaults to False.
        """
        super(LookingModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
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

    def forward_first_stage(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        return y
    
    def forward_second_stage(self, y):
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        #y = self.binarize(y)
        y = self.w2(y)
        #y = self.binarize(y)
        y = self.sigmoid(y)
        return y

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
    """
    Class definition of the Linear block
    """
    def __init__(self, linear_size=256, p_dropout=0.2):
        """
        Args:
            linear_size (int, optional): Size of the FC layers inside the block. Defaults to 256.
            p_dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(Linear, self).__init__()
        ###

        self.linear_size = linear_size
        self.p_dropout = p_dropout

        ###

        self.relu = nn.ReLU(True)
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
    """
    Class definition of the AlexNet crops model (Rasouli et al.).
    """
    def __init__(self, device, fine_tune=True):
        """
        Args:
            device (torch device): PyTorch device 
            fine_tune (bool, optional): Use the finetuned model. Defaults to True.
        """
        super(AlexNet_head, self).__init__()
        net = models.alexnet(pretrained=fine_tune).to(device)
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
            print('Fine Tune : ', fine_tune)
        self.net = net
    def forward(self, x):
        return self.net(x)

class ResNet18_head(nn.Module):
    def __init__(self, device, fine_tune=True):
        """
        Args:
            device (torch device): PyTorch device 
            fine_tune (bool, optional): Use the finetuned model. Defaults to True.
        """
        super(ResNet18_head, self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.net.fc  = nn.Sequential(
            nn.Linear(in_features=512, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device)
        if fine_tune:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.fc.parameters():
                param.requires_grad = True
        print('Fine Tune : ', fine_tune)

    def forward(self, x):
        return self.net(x)

class ResNet50_head(nn.Module):
    def __init__(self, device, fine_tune=True):
        """
        Args:
            device (torch device): PyTorch device 
            fine_tune (bool, optional): Use the finetuned model. Defaults to True.
        """
        super(ResNet50_head, self).__init__()
        net = models.resnext50_32x4d(pretrained=True)
        net.fc  = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device)
        if fine_tune:
            for param in net.parameters():
                param.requires_grad = False
            for param in net.fc.parameters():
                param.requires_grad = True
        self.net = net
        print('Fine Tune : ', fine_tune)

    def forward(self, x):
        return self.net(x)

class LookingNet_late_fusion_18(nn.Module):
    """
        Class definition for the combined Looking Model. Late fusion architecture with Resnet18 backbone. 
    """
    def __init__(self, PATH, PATH_look, device, fine_tune=True):
        """
        Args:
            PATH (str): Path to the pretrained ResNet18 heads model. Applicable only if fine-tune is enabled
            PATH_look (str): Path to the pretrained Looking joints model. Applicable only if fine-tune is enabled
            device (PyTorch device): PyTorch device
            fine_tune (bool, optional): Enable fine tune. Defaults to True.
        """
        super(LookingNet_late_fusion_18, self).__init__()
        self.backbone = ResNet18_head(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH))
            for m in self.backbone.net.parameters():
                m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(51)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()

        self.relu = nn.ReLU(inplace=True)

        self.encoder_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Linear(512, 1),
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
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))
        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        _ = self.backbone(head)
        _ = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), self.relu(layer_look)), 1).type(torch.float)
        return self.final(out_final)


class LookingNet_late_fusion_50(nn.Module):
    """
        Class definition for the combined Looking Model. Late fusion architecture with ResNext50 backbone. 
    """
    def __init__(self, PATH, PATH_look, device, fine_tune=True):
        """
        Args:
            PATH (str): Path to the pretrained ResNext50 heads model. Applicable only if fine-tune is enabled
            PATH_look (str): Path to the pretrained Looking joints model. Applicable only if fine-tune is enabled
            device (PyTorch device): PyTorch device
            fine_tune (bool, optional): Enable fine tune. Defaults to True.
        """
        super(LookingNet_late_fusion_50, self).__init__()
        self.backbone = ResNet50_head(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH))
            for m in self.backbone.net.parameters():
                m.requires_grad = False
            self.backbone.eval()

        self.looking_model = LookingModel(51)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()

        self.relu = nn.ReLU(inplace=True)

        self.encoder_head = nn.Sequential(
            nn.Linear(2048, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Linear(512, 1),
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
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))
        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        _ = self.backbone(head)
        _ = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), self.relu(layer_look)), 1).type(torch.float)
        return self.final(out_final)

class LookingNet_early_fusion_eyes(nn.Module):
    """
        Class definition for the combined Looking Model. Early fusion architecture with the eyes crops. 
    """
    def __init__(self, PATH_look, device, fine_tune=True):
        """
        Args:
            PATH_look (str): Path to the pretrained Looking joints model. Applicable only if fine-tune is enabled
            device (PyTorch device): PyTorch device
            fine_tune (bool, optional): Enable fine tune. Defaults to True.
        """
        super(LookingNet_early_fusion_eyes, self).__init__()
        self.looking_model = LookingModel(51)
        print("Fine tune : " , fine_tune)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()


        self.encoder_eyes = nn.Sequential(
            nn.Flatten(),
            nn.Linear(900, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )

        self.looking_module = Looking_early_module()
    def forward(self, x):
        
        eyes, keypoint = x
        encoded_eyes = self.encoder_eyes(eyes)

        output_first_stage = self.looking_model.forward_first_stage(keypoint)
        y = self.looking_module.forward(output_first_stage+encoded_eyes)
        return y

class Looking_early_module(nn.Module):
    def __init__(self, p_dropout=0.2, output_size=1, linear_size=256, num_stage=3):
        super(Looking_early_module, self).__init__()
        self.p_dropout = p_dropout
        self.linear_stages = []
        self.linear_size = linear_size
        self.output_size = output_size
        self.num_stage = num_stage

        for _ in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, y):
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        #y = self.binarize(y)
        y = self.w2(y)
        #y = self.binarize(y)
        y = self.sigmoid(y)
        return y


class LookingNet_early_fusion_18(nn.Module):
    """
        Class definition for the combined Looking Model. Early fusion architecture with ResNet18 backbone. 
    """
    def __init__(self, PATH, PATH_look, device, fine_tune=True):
        """
        Args:
            PATH (str): Path to the pretrained ResNet18 heads model. Applicable only if fine-tune is enabled
            PATH_look (str): Path to the pretrained Looking joints model. Applicable only if fine-tune is enabled
            device (PyTorch device): PyTorch device
            fine_tune (bool, optional): Enable fine tune. Defaults to True.
        """
        super(LookingNet_early_fusion_18, self).__init__()
        self.backbone = ResNet18_head(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            for m in self.backbone.net.parameters():
                m.requires_grad = False
            self.backbone.eval()

        self.looking_model = LookingModel(51)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )

        
        # post processing

        self.looking_module = Looking_early_module()

    def forward(self, x):
        head, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook
        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        _ = self.backbone(head)

        layer_resnet = activation["avgpool"]
        encoded_head = self.encoder_head(layer_resnet)

        output_first_stage = self.looking_model.forward_first_stage(keypoint)
        y = self.looking_module.forward(output_first_stage+encoded_head)
        return y

class LookingNet_early_fusion_50(nn.Module):
    """
        Class definition for the combined Looking Model. Early fusion architecture with ResNext50 backbone. 
    """
    def __init__(self, PATH, PATH_look, device, fine_tune=True):
        """
        Args:
            PATH (str): Path to the pretrained ResNext50 heads model. Applicable only if fine-tune is enabled
            PATH_look (str): Path to the pretrained Looking joints model. Applicable only if fine-tune is enabled
            device (PyTorch device): PyTorch device
            fine_tune (bool, optional): Enable fine tune. Defaults to True.
        """
        super(LookingNet_early_fusion_50, self).__init__()
        self.backbone = ResNet50_head(device)
        if fine_tune:
            self.backbone.load_state_dict(torch.load(PATH))
            for m in self.backbone.net.parameters():
                m.requires_grad = False
            self.backbone.eval()

        self.looking_model = LookingModel(51)
        if fine_tune:
            self.looking_model.load_state_dict(torch.load(PATH_look, map_location=torch.device(device)))
            for m in self.looking_model.parameters():
                m.requires_grad = False
            self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(2048, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )

        self.looking_module = Looking_early_module()

    def forward(self, x):
        head, keypoint = x
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook
        self.backbone.net.avgpool.register_forward_hook(get_activation('avgpool'))
        _ = self.backbone(head)

        layer_resnet = activation["avgpool"]
        encoded_head = self.encoder_head(layer_resnet)

        output_first_stage = self.looking_model.forward_first_stage(keypoint)
        y = self.looking_module(output_first_stage+encoded_head)
        #y = self.looking_model.forward_second_stage(output_first_stage+encoded_head)
        return y

class EyesModel(nn.Module):
    """
        Class definition for the Eyes model. It consists of a bunch of FC layers that takes as an input the flattened
        representation of the eyes crops.
    """
    def __init__(self, device):
        """
        Args:
            device (PyTorch device): PyTorch device
        """
        super(EyesModel, self).__init__()

        self.encoder_eyes = nn.Sequential(
            nn.Flatten(),
            nn.Linear(900, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        ).to(device)
    
        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded_eyes = self.encoder_eyes(x)
        return self.fc(encoded_eyes)
