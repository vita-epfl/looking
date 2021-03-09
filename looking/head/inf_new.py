from dataset import *
from utils import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
import argparse

torch.manual_seed(0)

# Parser 

parser = argparse.ArgumentParser(description='Training the head model on JAAD')

# parameters

parser.add_argument('-epochs', '--e', type=int, help='number of epochs for training', default=20)
parser.add_argument('-learning_rate', '--lr', type=float, help='learning rate for training', default=0.0001)
parser.add_argument('-split', '--s', type=str, help='dataset split', default="original")
parser.add_argument('-model', '--m', type=str, help='model type [resnet18, resnet50, alexnet]', default="resnet50")
parser.add_argument('-path', '--pt', type=str, help='path for model saving', default='./models/')
parser.add_argument('-save', help='save the model', action='store_true')

args = parser.parse_args()


EPOCHS = args.e
split = args.s
model_type = args.m

assert model_type in ['resnet18', 'resnet50', 'alexnet']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)
res_18 =False

if model_type == "alexnet":
	data_transform = transforms.Compose([
		SquarePad(),
        transforms.ToTensor(),
	transforms.ToPILImage(),
        transforms.Resize((227,227)),
	transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
	])
	model = models.alexnet(pretrained=True).to(device)
	model.classifier  = nn.Sequential(
            	nn.Dropout(),
            	nn.Linear(256 * 6 * 6, 4096),
            	nn.ReLU(inplace=True),
            	nn.Dropout(),
            	nn.Linear(4096, 4096),
            	nn.ReLU(inplace=True),
            	nn.Linear(4096, 1),
        	    nn.Sigmoid()
	).to(device)
	for param in model.parameters():
        	param.requires_grad = False


	for param in model.classifier.parameters():
	        param.requires_grad = True
else:
	data_transform = transforms.Compose([
		SquarePad(),
        transforms.ToTensor(),
	transforms.ToPILImage(),
        transforms.Resize((224,224)),
	transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
	])
	if model_type == "resnet18":
		model = models.resnet18(pretrained=True)
		model.fc  = nn.Sequential(
			nn.Linear(in_features=512, out_features=1, bias=True),
			nn.Sigmoid()
		).to(device)
	elif model_type == "resnet50":
		model = models.resnext50_32x4d(pretrained=True)
		model.fc  = nn.Sequential(
			nn.Linear(in_features=2048, out_features=1, bias=True),
			nn.Sigmoid()
		).to(device)

print("model type {} | split type : {}".format(model_type, split))

model.load_state_dict(torch.load('{}{}_head_{}.p'.format(args.pt, model_type, split)))
model.eval()

jaad_test = JAAD_Dataset_head_new("../data/", "JAAD_2k30/", "test", split, data_transform)
ap, acc = jaad_test.evaluate(model.to(device), device, 10)

print("AP JAAD : {} | Acc JAAD : {}".format(ap, acc))

data_test = Kitti_Dataset_head("test", data_transform)

dataset_loader_test = torch.utils.data.DataLoader(data_test,batch_size=8, shuffle=True)

acc = 0
ap = 0
out_lab = torch.Tensor([]).type(torch.float).to(device)
test_lab = torch.Tensor([]).to(device)
for x_test, y_test in dataset_loader_test:
    if use_cuda:
        x_test, y_test = x_test.cuda(), y_test.cuda()
    output = model(x_test)
    #out_pred, pred_label = torch.max(nn.Softmax(dim=-1)(output), dim=-1)
    #ut_pred, pred_label = torch.max(output, dim=-1)
    out_pred = output
    le = x_test.shape[0]
    pred_label = torch.round(out_pred)
    #acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
    #ap += le*average_precision(out_pred, y_test)
    test_lab = torch.cat((test_lab.detach().cpu(), y_test.detach().cpu().view(-1)), dim=0)
    #out_lab = torch.cat((out_lab.detach().cpu(), nn.Sigmoid(out_pred.detach().cpu()).view(-1)), dim=0)
    out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)

ap = average_precision(out_lab, test_lab)
acc = binary_acc(torch.round(out_lab).type(torch.float).view(-1), test_lab).item()
print('Kitti {} | acc:{} | ap:{}'.format(0, acc,ap))
