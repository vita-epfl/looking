from utils import *
from dataset import *
from network import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import datasets, models, transforms
import argparse

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

assert model_type in ['resnet18', 'resnet50', 'alexnet']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
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

if model_type=='resnet18':
    model = LookingNet_early_fusion_18("../head/models/resnet18_head_{}.p".format(split), "../joints/models/looking_model_jaad_{}_full_kps.p".format(split), device).to(device)
elif model_type=='resnet50':
    model = LookingNet_early_fusion_50("../head/models/resnet50_head_{}.p".format(split), "../joints/models/looking_model_jaad_{}_full_kps.p".format(split), device).to(device)

model.load_state_dict(torch.load("./models/model_combined_{}_{}.p".format(model_type, split)))
model.eval()

model.cuda()

data_test_jaad = JAAD_Dataset_new('../data/', 'JAAD_2k30/', "test", split, data_transform)

data_test= Kitti_Dataset("test", data_transform)
dataset_loader_test = torch.utils.data.DataLoader(data_test,batch_size=8, shuffle=True)

acc = 0
ap = 0

out_lab = torch.Tensor([]).type(torch.float).to(device)
test_lab = torch.Tensor([]).to(device)
for heads, keypoints, y_test in dataset_loader_test:
    if use_cuda:
        heads, keypoints, y_test = heads.cuda(), keypoints.cuda(), y_test.cuda()
    output = model(heads, keypoints)
    out_pred = output
    le = heads.shape[0]
    pred_label = torch.round(out_pred)
    test_lab = torch.cat((test_lab.detach().cpu(), y_test.detach().cpu().view(-1)), dim=0)
    out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)

ap = average_precision(out_lab, test_lab)
acc = binary_acc(torch.round(out_lab).type(torch.float).view(-1), test_lab).item()
print('Performance on Kitti | acc:{} | ap:{}'.format(acc,ap))
ap, acc = data_test_jaad.evaluate(model, device, 10)
print('Performance on JAAD | acc:{} | ap:{}'.format(acc,ap))

