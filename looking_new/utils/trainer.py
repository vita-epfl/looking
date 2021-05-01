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
        optimizer_type = self.general['optimizer']
        self.data_transform = None
        self.grad_map = False
        assert criterion_type in ['BCE', 'focal_loss']
        assert optimizer_type in ['adam', 'sgd']
        if criterion_type == 'BCE':
            criterion = nn.BCELoss()
        else:
            criterion = FocalLoss(alpha=1, gamma=3)

        
        
        model_type = self.model_type['type']
        pose = self.general['pose']

        assert model_type in ['joints', 'heads', 'heads+joints']
        assert pose in ['head', 'body', 'full']
        if model_type == 'joints':
            self.grad_map = self.general.getboolean('grad_map')
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
        self.model_type_ = model_type
        self.pose = pose
        self.lr = float(self.general['learning_rate'])
        self.epochs = int(self.general['epochs'])
        self.batch_size = int(self.general['batch_size'])
        if optimizer_type == 'adam':
	        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        return model, criterion, optimizer, self.data_transform

    def get_data(self):
        data_type = self.data_args['name']
        split_strategy = self.data_args['split']
        path_txt = self.data_args['path_txt']
        dataset_train = None

        if data_type == 'JAAD':
            path_data = self.data_args['path_data']
            dataset_train = JAAD_Dataset(path_data, self.model_type_, 'train', self.pose, split_strategy, self.data_transform, path_txt)
            dataset_val = JAAD_Dataset(path_data, self.model_type_, 'val', self.pose, split_strategy, self.data_transform, path_txt)
        return dataset_train, dataset_val

    def parse(self):
        self.model, self.criterion, self.optimizer, self.data_transform = self.get_model()
        self.dataset_train, self.dataset_val = self.get_data()
        self.path_output = os.path.join(self.general['path'], self.model_type['type'].title())
        try:
            os.makedirs(self.path_output)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        if self.model_type['type'] != 'joints':
            name_model = '_'.join([self.model.__class__.__name__, self.criterion.__class__.__name__])+'.p'
        else:
            name_model = '_'.join([self.model.__class__.__name__, self.criterion.__class__.__name__, self.general['pose']])+'.p'
        self.path_model = os.path.join(self.path_output, name_model)

class Trainer():
    """
        Class definition for training and saving the trained model
    """
    def __init__(self, parser):
        self.parser = parser
    
    def train(self):
        self.parser.model = self.parser.model.train()
        train_loader = DataLoader(self.parser.dataset_train, batch_size=self.parser.batch_size, shuffle=True)
        running_loss = 0
        i = 0
        best_ap = 0
        for epoch in range(self.parser.epochs):
            losses = []
            accuracies = []

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.parser.device), y_batch.to(self.parser.device)
                
                self.parser.optimizer.zero_grad()
                output = self.parser.model(x_batch)
                loss = self.parser.criterion(output, y_batch.view(-1, 1).float())
                running_loss += loss.item()



                loss.backward()
                losses.append(loss.item())
                accuracies.append(binary_acc(output.type(torch.float).view(-1), y_batch).item())

                self.parser.optimizer.step()
                i += 1

                if i%10 == 0:
                    print('step {} , loss :{} | acc:{} '.format(i, np.mean(losses), np.mean(accuracies)))
                    losses = []
                    accuracies = []
            i = 0
            best_ap, ap_val, acc_val = self.eval_epoch(best_ap)
            print('Epoch {} | mAP : {} | mAcc :{}'.format(epoch+1, ap_val, acc_val))

    
    def eval_epoch(self, best_ap):
        aps, accs = self.parser.dataset_val.evaluate(self.parser.model, self.parser.device, it=1)
        if aps > best_ap:
            best_ap = aps
            torch.save(self.parser.model, self.parser.path_model)
        return best_ap, aps, accs