from dataset import *
from utils import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
import argparse
from network import *


torch.manual_seed(0)

# Parser

parser = argparse.ArgumentParser(description='Training the eyes model on JAAD')

# parameters

parser.add_argument('--epochs', '-e', type=int, help='number of epochs for training', default=100)
parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate for training', default=0.0001)
parser.add_argument('--path', type=str, help='path for model saving', default='./models/')
parser.add_argument('--jaad_split_path', type=str, help='proportion for the training', default="JAAD_2k30_eyes/")
parser.add_argument('--split_path', '-jsp', type=str, help='proportion for the training', default="/home/caristan/looking/looking/splits/")
parser.add_argument('--data_path', '-dp', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/data/")


args = parser.parse_args()

EPOCHS = args.epochs
LEARNING_RATE= args.learning_rate

DATA_PATH = args.data_path
SPLIT_PATH = args.split_path
JAAD_PATH = args.jaad_split_path
PATH_MODEL = args.path


"""
My local paths
DATA_PATH = '../../data/'
SPLIT_PATH_JAAD = '../splits/'
PATH_MODEL = './models/'
"""

# Resize
resize = (10, 15)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)

data_transform = transforms.Compose([
	SquarePad(),
    transforms.ToTensor(),
	transforms.ToPILImage(),
    transforms.Resize(resize),
	transforms.ToTensor(),
])


INPUT_SIZE = 450

model = LookingModel(INPUT_SIZE).to(device)

jaad_train = JAAD_Dataset(DATA_PATH, JAAD_PATH, "train", SPLIT_PATH, data_transform)
jaad_val = JAAD_Dataset(DATA_PATH, JAAD_PATH, "val", SPLIT_PATH, data_transform)

dataset_loader = torch.utils.data.DataLoader(jaad_train, batch_size=64, shuffle=True)
dataset_loader_test = torch.utils.data.DataLoader(jaad_val, batch_size=8, shuffle=True)

best_ap, best_ac, best_epoch = 0, 0, 0

"""
criterion = nn.BCELoss()
test_ap = []
test_ac = []
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.train()
for epoch in range(EPOCHS):
	i = 0
	model.to(device)
	running_loss = 0
	for x_batch, y_batch in dataset_loader:
		if torch.cuda.is_available():
			x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
		optimizer.zero_grad()
		output = model(x_batch.float())
		loss = criterion(output, y_batch.view(-1, 1))
		running_loss += loss.item() * x_batch.size(0)
		loss.backward()
		optimizer.step()
		i+=1
		if i%10 == 0:
			model.eval()
			out_pred = output
			pred_label = torch.round(out_pred)
			acc = binary_acc(pred_label.type(torch.float).view(-1), y_batch).item()
			print('step {} , loss :{} | acc:{} '.format(i, loss.item(), acc))
			model.train()
			#break
	model.eval()
	torch.cuda.empty_cache()

	ap_test, acc_test = jaad_val.evaluate(model, device, 1)

	test_ap.append(ap_test)
	test_ac.append(acc_test)

	model.train()
	if ap_test > best_ap:
		best_epoch = epoch
		best_ap = ap_test
		best_ac = acc_test
		print(f'Best epoch: {best_epoch}, AP: {best_ap}, Acc: {best_ac}')
		torch.save(model, PATH_MODEL + 'eyes_model.pkl')

"""

model = LookingModel(INPUT_SIZE).to(device)
model.load_state_dict(torch.load('eyes_model.pkl'))
model.eval()
jaad_test = JAAD_Dataset(DATA_PATH, JAAD_PATH, "test", SPLIT_PATH, data_transform)
ap, acc = jaad_test.evaluate(model, device, 1)
print("AP JAAD : {} | Acc JAAD : {}".format(ap, acc))
