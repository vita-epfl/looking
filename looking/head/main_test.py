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
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)
res_18 = False

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
	for param in net.parameters():
        	param.requires_grad = False


	for param in net.classifier.parameters():
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
		net = models.resnet18(pretrained=True)
		net.fc  = nn.Sequential(
			nn.Linear(in_features=512, out_features=1, bias=True),
			nn.Sigmoid()
		).to(device)
	elif model_type == "resnet50":
		net = models.resnext50_32x4d(pretrained=True)
		net.fc  = nn.Sequential(
			nn.Linear(in_features=2048, out_features=1, bias=True),
			nn.Sigmoid()
		).to(device)

print("model type {} | split type : {}".format(model_type, split))

jaad_train = JAAD_Dataset_head_new("/home/caristan/code/looking/looking/data/", "JAAD_2k30/", "train", split, data_transform)
jaad_val = JAAD_Dataset_head_new("/home/caristan/code/looking/looking/data/", "JAAD_2k30/", "val", split, data_transform)

dataset_loader = torch.utils.data.DataLoader(jaad_train, batch_size=64, shuffle=True)
dataset_loader_test = torch.utils.data.DataLoader(jaad_val, batch_size=8, shuffle=True)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

i = 0
aps_val = 0
accs_val = 0

if use_cuda:
	net.to(device)

for e in range(EPOCHS):
	i = 0
	for x_batch, y_batch in dataset_loader:
		if use_cuda:
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)
		optimizer.zero_grad()
		output = net(x_batch)
		l = loss(output.view(-1), y_batch.type(torch.float).view(-1))#.cuda()
		l.backward()
		optimizer.step()
		i += 1

		if i%10 == 0:
			net.eval()
			out_pred = output
			pred_label = torch.round(out_pred)
			acc = binary_acc(pred_label.type(torch.float).view(-1), y_batch).item()
			print('step {} , loss :{} | acc:{} '.format(i, l.item(), acc))
			net.train()
			#break
	net.eval()
	torch.cuda.empty_cache()
	"""
	acc = 0
	ap = 0
	out_lab = torch.Tensor([]).type(torch.float)
	test_lab = torch.Tensor([])
	for x_test, y_test in dataset_loader_test:
		if use_cuda:
			x_test, y_test = x_test.to(device), y_test.to(device)
		output = net(x_test)
		out_pred = output
		pred_label = torch.round(out_pred)
		le = x_test.shape[0]
		acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
		test_lab = torch.cat((test_lab.detach().cpu(), y_test.view(-1).detach().cpu()), dim=0)
		out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
	acc /= len(jaad_val)
	ap = average_precision(out_lab, test_lab)
	"""
	ap, ac = jaad_val.evaluate(net, device, 1)
	if ap > aps_val:
		accs_val = acc
		aps_val = ap
		if args.save:
			torch.save(net.state_dict(),'/home/caristan/code/looking/looking/joints/models/' + '{}{}_head_{}_romain.p'.format(args.pt, model_type, split))
	print('epoch {} | acc:{} | ap:{}'.format(e, acc, ap))
	net.train()
