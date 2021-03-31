from utils import *
from dataset import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)

data_transform = transforms.Compose([
        SquarePad(),
        transforms.ToTensor(),
    transforms.ToPILImage(),
        transforms.Resize((224,224)),
    transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             #   std=[1, 1, 1])
                             std=[0.229, 0.224, 0.225])
])
"""
model = models.alexnet(pretrained=True)
model.classifier  = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
            nn.Sigmoid()
)
"""


"""
model = models.vgg16(pretrained=True)
#exit(0)
model.classifier  = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1),
            nn.Sigmoid()
)
"""
video = False
original = True

model = models.resnext50_32x4d(pretrained=True)
model.fc  = nn.Sequential(
    nn.Linear(in_features=2048, out_features=1, bias=True),
    nn.Sigmoid()
).to(device)


"""
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
	nn.Linear(in_features=512, out_features=1, bias=True),
	nn.Sigmoid()
).to(device)
"""
model.load_state_dict(torch.load("./models/resnet50_head_original.p"))
model.eval()

model.cuda()

#data_test = JAAD_Dataset_head("test", video, original, data_transform)
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
print('epoch {} | acc:{} | ap:{}'.format(0, acc,ap))
