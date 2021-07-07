from dataset import *
from utils import *
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
import argparse

torch.manual_seed(0)

# Parser

parser = argparse.ArgumentParser(description='Analysis of the head model on JAAD')

# parameters

parser.add_argument('--model', '-m', type=str, help='model type [resnet18, resnet50, alexnet]', default="resnet50")
parser.add_argument('--split', type=str, help='dataset split', default="video")
parser.add_argument('--path', type=str, help='path for model saving', default='./models/')
parser.add_argument('--jaad_split_path', '-jsp', type=str, help='proportion for the training', default="JAAD_2k30/")
parser.add_argument('--split_path', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/splits/")
parser.add_argument('--data_path', '-dp', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/data/")


args = parser.parse_args()

split = args.split
model_type = args.model

DATA_PATH = args.data_path
SPLIT_PATH = args.split_path
JAAD_PATH = args.jaad_split_path
PATH_MODEL = args.path


My local paths
DATA_PATH = '../../data/'
SPLIT_PATH_JAAD = '../splits/'
PATH_MODEL = './models/'


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

model.load_state_dict(torch.load('{}{}_head_{}_new_crops.p'.format(PATH_MODEL, model_type, split)))
model.eval()

jaad_test = JAAD_Dataset_head_test(DATA_PATH, JAAD_PATH, "test", SPLIT_PATH, split, data_transform)
ap, acc, f_pos, f_neg = jaad_test.get_mislabeled_test(model, device)

"""with open("false_pos_head_new_crops.txt", "w") as output:
    for row in f_pos:
        output.write(str(row)+'\n')
with open("false_neg_head_new_crops.txt", "w") as output:
    for row in f_neg:
        output.write(str(row)+'\n')

print("AP JAAD : {} | Acc JAAD : {}".format(ap, acc))
print(f"False positives: {len(f_pos)}, False negatives: {len(f_neg)}")
"""
