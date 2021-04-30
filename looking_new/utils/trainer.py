import configparser
from utils.dataset import *
from utils.network import *
from utils.losses import *
import os, errno

class Parser():
    """
        Class definition for parser in order to get the right arguments
    """
    def __init__(self, config):
        self.general = config['General']
        self.model_type = config['Model_type']
        self.eval_params = config['Eval']
        self.data_args = config['Dataset']
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(self.general['device']) if use_cuda else "cpu")
        print('Device: ', self.device)
    
    def get_model(self):
        criterion_type = self.general['loss']
        self.data_transform = None
        assert criterion_type in ['BCE', 'focal_loss']
        if criterion_type == 'BCE':
            criterion = nn.BCELoss()
        else:
            criterion = FocalLoss(alpha=1, gamma=3)
        
        model_type = self.model_type['type']
        pose = self.general['pose']

        assert model_type in ['joints', 'heads', 'heads+joints']
        assert pose in ['head', 'body', 'full']
        if model_type == 'joints':
            if pose == "head":
                INPUT_SIZE = 15
            elif pose == "body":
                INPUT_SIZE = 36
            else:
                INPUT_SIZE = 51
            model = LookingModel(INPUT_SIZE).to(self.device)
        elif model_type == 'heads':
            backbone = self.model_type['backbone']
            fine_tune = self.model_type.getboolean('fine_tune')
            assert backbone in ['alexnet', 'resnet18', 'resnet50']
            if backbone == 'alexnet':
                model = AlexNet_head(self.device, fine_tune)
                self.data_transform = transforms.Compose([
                        SquarePad(),
                        transforms.ToTensor(),
                    transforms.ToPILImage(),
                        transforms.Resize((227,227)),
                    transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
            elif backbone == 'resnet18':
                model = ResNet18_head(self.device)
                self.data_transform = transforms.Compose([
                        SquarePad(),
                        transforms.ToTensor(),
                    transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                    transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])
            else:
                model = ResNet50_head(self.device)
                self.data_transform = transforms.Compose([
                        SquarePad(),
                        transforms.ToTensor(),
                    transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                    transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])
        return model, criterion, self.data_transform


    def parse(self):
        self.model, self.criterion, self.data_transform = self.get_model()
        self.path_output = os.path.join(self.general['path'], self.model_type['type'].title())
        try:
            os.makedirs(self.path_output)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        if self.model_type['type'] != 'joints':
            name_model = '_'.join([model.__class__.__name__, criterion.__class__.__name__])+'.p'
        else:
            name_model = '_'.join([model.__class__.__name__, criterion.__class__.__name__, self.general['pose']])+'.p'
        self.path_model = os.path.join(self.path_output, name_model)

class Trainer():
    """
        Class definition for training and saving model
    """