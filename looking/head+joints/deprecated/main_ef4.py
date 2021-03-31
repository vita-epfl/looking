from dataset import *
from utils import *
from network import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, dataset_loader_val, data_test):
	model.eval()
	acc = 0
	ap = 0
	out_lab = torch.Tensor([]).type(torch.float)
	test_lab = torch.Tensor([])
	
	for heads, keypoints, y_batch in dataset_loader_val:
		if use_cuda:
			heads, keypoints, y_batch = heads.to(device), keypoints.to(device), y_batch.to(device)
		output = model(heads, keypoints)
		le = output.shape[0]
		out_pred = output
		pred_label = torch.round(out_pred)
		acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_batch).item()

		#ap += le*average_precision(out_pred, y_test)
		test_lab = torch.cat((test_lab.detach().cpu(), y_batch.view(-1).detach().cpu()), dim=0)
		out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
	#print(out_lab.shape)
	acc /= len(data_test)
	#ap /= len(data_test)
	ap = average_precision(out_lab, test_lab)
	return acc, ap

EPOCHS = 20
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print('Device: ', device)

data_transform = transforms.Compose([
		SquarePad(),
        transforms.ToTensor(),
	transforms.ToPILImage(),
        transforms.Resize((224,224)),
	transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

video = False
jaad_train = JAAD_Dataset("train", video, data_transform)
jaad_val = JAAD_Dataset("val", video, data_transform)

data_test = Kitti_Dataset("test", data_transform)


model = LookingNet_early_fusion_5("../head/models/resnet_head_best.p", "../joints/models/looking_model_jaad.p", device).to(device)

#exit(0)

dataset_loader = torch.utils.data.DataLoader(jaad_train, batch_size=64, shuffle=True)
dataset_loader_val = torch.utils.data.DataLoader(jaad_val, batch_size=32, shuffle=True)
dataset_loader_kitti = torch.utils.data.DataLoader(data_test, batch_size=16, shuffle=True)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(list(model.encoder_head.parameters()) + list(model.final.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.0005)

#print(model)
i = 0
accs = []
losses = []
ap_max = 0
writer = SummaryWriter(log_dir="./runs7")
for e in range(EPOCHS):
	for heads, keypoints, y_batch in dataset_loader:
		if use_cuda:
			heads, keypoints, y_batch = heads.to(device), keypoints.to(device), y_batch.to(device)
		optimizer.zero_grad()
		output = model(heads, keypoints)
		l = loss(output.view(-1), y_batch.type(torch.float).view(-1)).to(device)
		#for w in model.final.parameters():
			#print(w)
		#	l += w.norm(1)
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
			writer.add_scalar('Loss/train', np.mean(losses), i)
   			#writer.add_scalar('Loss/test', np.random.random(), i)
			writer.add_scalar('Accuracy/train', np.mean(accs), i)
			accs = []
			losses = []
	model.eval()
	torch.cuda.empty_cache() 
	acc = 0
	ap = 0
	out_lab = torch.Tensor([]).type(torch.float)
	test_lab = torch.Tensor([])
	
	for heads, keypoints, y_batch in dataset_loader_val:
		if use_cuda:
			heads, keypoints, y_batch = heads.to(device), keypoints.to(device), y_batch.to(device)
		output = model(heads, keypoints)
		le = output.shape[0]
		out_pred = output
		pred_label = torch.round(out_pred)
		acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_batch).item()

		#ap += le*average_precision(out_pred, y_test)
		test_lab = torch.cat((test_lab.detach().cpu(), y_batch.view(-1).detach().cpu()), dim=0)
		out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
	#print(out_lab.shape)
	acc /= len(jaad_val)
	#ap /= len(data_test)
	ap = average_precision(out_lab, test_lab)
	print('epoch {} | acc:{} | ap:{}'.format(e+1, acc,ap))
	writer.add_scalar('AP/val', ap, e)
   	#writer.add_scalar('Loss/test', np.random.random(), i)
	writer.add_scalar('Accuracy/val', acc, e)

	acc_kitti, ap_kitti = evaluate(model, dataset_loader_kitti, data_test)
	writer.add_scalar('AP/test_Kitti', ap_kitti, e)
   	#writer.add_scalar('Loss/test', np.random.random(), i)
	writer.add_scalar('Accuracy/test_Kitti', acc_kitti, e)

	if ap > ap_max:
		ap_max = ap
		torch.save(model.state_dict(), "./models/model_early_stop_6.p")
	model.train()
	#break
