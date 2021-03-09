from dataset import *
from utils import *
from network import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse

torch.manual_seed(0)

# Parser 

parser = argparse.ArgumentParser(description='Training the head model on JAAD')

# parameters

parser.add_argument('--epochs', '-e', type=int, help='number of epochs for training', default=20)
parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate for training', default=0.0001)
parser.add_argument('--split', '-s', type=str, help='dataset split', default="original")
parser.add_argument('--model', '-m', type=str, help='model type [resnet18, resnet50]', default="resnet18")
parser.add_argument('--path', '-pt', type=str, help='path for model saving', default='./models/')

args = parser.parse_args()


EPOCHS = args.epochs
split = args.split
model_type = args.model
video = args.split

assert model_type in ['resnet18', 'resnet50']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print('Device: ', device)


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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print('Device: ', device)



jaad_train = JAAD_Dataset_new('../data/', 'JAAD_2k30/',"train", split, data_transform)
jaad_val = JAAD_Dataset_new('../data/', 'JAAD_2k30/', "val", split, data_transform)


if model_type=='resnet18':
    model = LookingNet_early_fusion_18("../head/models/resnet18_head_{}.p".format(split), "../joints/models/looking_model_jaad_{}_full_kps.p".format(split), device).to(device)
elif model_type=='resnet50':
    model = LookingNet_early_fusion_50("../head/models/resnet50_head_{}.p".format(split), "../joints/models/looking_model_jaad_{}_full_kps.p".format(split), device).to(device)
#model = LookingNet_early_fusion_18("../head/models/resnet18_head_video.p", "../joints/models/looking_model_jaad_video_full_kps.p", device).to(device)


dataset_loader = torch.utils.data.DataLoader(jaad_train, batch_size=64, shuffle=True)
dataset_loader_val = torch.utils.data.DataLoader(jaad_val, batch_size=32, shuffle=True)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(list(model.encoder_head.parameters()) + list(model.final.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.0005)

#print(model)
i = 0
accs = []
losses = []
ap_max = 0

for e in range(EPOCHS):
	for heads, keypoints, y_batch in dataset_loader:
		if use_cuda:
			heads, keypoints, y_batch = heads.to(device), keypoints.to(device), y_batch.to(device)
		optimizer.zero_grad()
		output = model(heads, keypoints)
		l = loss(output.view(-1), y_batch.type(torch.float).view(-1)).to(device)
		l.backward()
		optimizer.step()
		i += 1
		model.eval()
		pred_label = torch.round(output)
		accs.append(binary_acc(pred_label.type(torch.float).view(-1), y_batch).item())
		losses.append(l.item())
		model.train()
		if i%10 == 0:
			print('step {} , m_loss :{} | m_acc:{} '.format(i, np.mean(losses), np.mean(accs)))
			accs = []
			losses = []
			#break
	model.eval()
	torch.cuda.empty_cache() 
	acc = 0
	ap = 0
	out_lab = torch.Tensor([]).type(torch.float)
	test_lab = torch.Tensor([])

	ap, acc = jaad_val.evaluate(model, device, 1)

	print('epoch {} | acc:{} | ap:{}'.format(e+1, acc,ap))

	if ap > ap_max:
		ap_max = ap
		torch.save(model.state_dict(), "./models/model_combined_{}_{}.p".format(model_type, video))
	model.train()
	#break
