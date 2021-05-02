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
        self.eval_it = int(self.general['eval_it'])
        self.dropout = float(self.general['dropout'])
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
            model = LookingModel(INPUT_SIZE, self.dropout).to(self.device)
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
        else:
            backbone = self.model_type['backbone']
            fine_tune = self.model_type.getboolean('fine_tune')
            self.data_transform = transforms.Compose([
                        SquarePad(),
                        transforms.ToTensor(),
                    transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                    transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])
            assert backbone in ['resnet18', 'resnet50']
            name_model_joints = '_'.join(['LookingModel', criterion.__class__.__name__, self.general['pose']])+'.p'
            if backbone == 'resnet18':
                name_model_backbone = '_'.join(['ResNet18_head', criterion.__class__.__name__])+'.p'
                
            else:
                name_model_backbone = '_'.join(['ResNet50_head', criterion.__class__.__name__])+'.p'

            path_output_model_backbone = os.path.join(self.general['path'], 'Heads')
            path_backbone = os.path.join(path_output_model_backbone, name_model_backbone)

            path_output_model_joints = os.path.join(self.general['path'], 'Joints')
            path_model_joints = os.path.join(path_output_model_joints, name_model_joints)
            if fine_tune:
                if not os.path.isfile(path_backbone):
                    print('ERROR: Heads model not trained, please train your heads model first')
                    exit(0)
                if not os.path.isfile(path_model_joints):
                    print('ERROR: Joints model not trained, please train your joints model first')
                    exit(0)
                
            if backbone == 'resnet18':
                model = LookingNet_early_fusion_18(path_backbone, path_output_model_joints, self.device, fine_tune)
            else:
                model = LookingNet_early_fusion_50(path_backbone, path_output_model_joints, self.device, fine_tune)
            


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

    def get_data(self, data_type):
        split_strategy = self.data_args['split']
        path_txt = os.path.join(self.data_args['path_txt'], 'splits_'+data_type.lower())
        dataset_train = []
        dataset_val = []

        if data_type == 'JAAD':
            path_data = self.data_args['path_data']
            dataset_train = JAAD_Dataset(path_data, self.model_type_, 'train', self.pose, split_strategy, self.data_transform, path_txt, self.device)
            dataset_val = JAAD_Dataset(path_data, self.model_type_, 'val', self.pose, split_strategy, self.data_transform, path_txt, self.device)
        elif data_type == 'Kitti':
            path_data = self.data_args['path_data']
            dataset_train = Kitti_dataset('train', self.model_type_, path_txt, path_data, self.pose, self.data_transform, self.device)
            dataset_val = Kitti_dataset('val', self.model_type_, path_txt, path_data, self.pose, self.data_transform, self.device)
        return dataset_train, dataset_val
    
    def get_data_test(self, data_type):
        split_strategy = self.eval_params['split']
        path_txt = os.path.join(self.data_args['path_txt'], 'splits_'+data_type.lower())
        dataset_test = []

        if data_type == 'JAAD':
            path_data = self.eval_params['path_data_eval']
            dataset_test = JAAD_Dataset(path_data, self.model_type_, 'test', self.pose, split_strategy, self.data_transform, path_txt, self.device)
        elif data_type == 'Kitti':
            path_data = self.eval_params['path_data_eval']
            dataset_test = Kitti_dataset('test', self.model_type_, path_txt, path_data, self.pose, self.data_transform, self.device)
        return dataset_test

    def parse(self):
        self.model, self.criterion, self.optimizer, self.data_transform = self.get_model()
        self.dataset_train, self.dataset_val = self.get_data(self.data_args['name'])
        self.path_output = os.path.join(self.general['path'], self.data_args['name'], self.model_type['type'].title())
        try:
            os.makedirs(self.path_output)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        additional_features = ''
        if 'JAAD' in self.data_args['name']:
            additional_features += '{}'.format(self.data_args['split'])

        if self.model_type['type'] != 'joints':
            name_model = '_'.join([self.model.__class__.__name__, self.criterion.__class__.__name__, additional_features])+'.p'
        else:
            name_model = '_'.join([self.model.__class__.__name__, self.criterion.__class__.__name__, self.general['pose'], additional_features])+'.p'
        self.path_model = os.path.join(self.path_output, name_model)
    
    def load_model_for_eval(self):
        self.model.load_state_dict(torch.load(self.path_model))
        self.model = self.model.to(self.device).eval()

class Evaluator():
    """
        Class definition for evaluation. To run once you have the traind model
    """
    def __init__(self, parser):
        self.parser = parser
        if os.path.isfile(self.parser.path_model):
            print('Model file exists.. Loading model file ...')
            self.parser.load_model_for_eval()
        else:
            print('ERROR : Model file doesnt exists, please train your model first or review your parameters')
            exit(0)
    def evaluate(self):
        """
            Loop over the test set and evaluate the performance of the model on it
        """
        data_to_evaluate = self.parser.eval_params['eval_on']
        data_test = self.parser.get_data_test(data_to_evaluate)
        data_loader_test = DataLoader(data_test, 1, shuffle=False)
        if data_to_evaluate != 'JAAD':
            acc = 0
            ap = 0

            output_all = torch.Tensor([]).type(torch.float).to(self.parser.device)
            labels_all = torch.Tensor([]).to(self.parser.device)
            for x_batch, y_batch in data_loader_test:

                y_batch = y_batch.to(self.parser.device)
                output = self.parser.model(x_batch)

                pred_label = torch.round(output)

                labels_all = torch.cat((labels_all.detach().cpu(), y_batch.detach().cpu().view(-1)), dim=0)
                output_all = torch.cat((output_all.detach().cpu(), output.view(-1).detach().cpu()), dim=0)

            ap = average_precision(output_all, labels_all)
            acc = binary_acc(output_all.type(torch.float).view(-1), labels_all).item()
        else:
            ap, acc = data_test.evaluate(self.parser.model, self.parser.device, 10)
        print('Evaluation on {} | acc:{:.1f} | ap:{:.1f}'.format(data_to_evaluate, acc, ap*100))
        


class Trainer():
    """
        Class definition for training and saving the trained model
    """
    def __init__(self, parser):
        self.parser = parser
    
    def train(self):
        self.parser.model = self.parser.model.to(self.parser.device).train()
        train_loader = DataLoader(self.parser.dataset_train, batch_size=self.parser.batch_size, shuffle=True)
        running_loss = 0
        i = 0
        best_ap = 0
        for epoch in range(self.parser.epochs):
            losses = []
            accuracies = []

            for x_batch, y_batch in train_loader:
                y_batch = y_batch.to(self.parser.device)
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
                    print_summary_step(i, np.mean(losses), np.mean(accuracies))
                    #print('step {} , loss :{} | acc:{} '.format(i, np.mean(losses), np.mean(accuracies)))
                    losses = []
                    accuracies = []
                    
            i = 0
            best_ap, ap_val, acc_val = self.eval_epoch(best_ap)
            print('')
            print('Epoch {} | mAP_val : {} | mAcc_val :{}'.format(epoch+1, ap_val, acc_val))

    
    def eval_epoch(self, best_ap):
        aps, accs = self.parser.dataset_val.evaluate(self.parser.model, self.parser.device, it=self.parser.eval_it)
        if aps > best_ap:
            best_ap = aps
            torch.save(self.parser.model.state_dict(), self.parser.path_model)
        return best_ap, aps, accs