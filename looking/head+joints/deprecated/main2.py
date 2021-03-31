from dataset import *
from utils import *
from network import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F


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

model = LookingNet9("../head/models/resnet_head_best.p")

dataset_loader = torch.utils.data.DataLoader(jaad_train, batch_size=128, shuffle=True)
dataset_loader_val = torch.utils.data.DataLoader(jaad_val, batch_size=32, shuffle=True)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.0005)

#print(model)
i = 0
accs = []
losses = []
ap_max = 0
for e in range(EPOCHS):
	for heads, keypoints, y_batch in dataset_loader:
		optimizer.zero_grad()
		output = model(heads, keypoints)
		l = loss(output.view(-1), y_batch.type(torch.float).view(-1)).cuda()
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
	model.eval()
	torch.cuda.empty_cache() 
	acc = 0
	ap = 0
	out_lab = torch.Tensor([]).type(torch.float)
	test_lab = torch.Tensor([])
	
	for heads, keypoints, y_batch in dataset_loader_val:
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
	if ap > ap_max:
		ap_max = ap
		torch.save(model.state_dict(), "./models/model9.p")
	#break
