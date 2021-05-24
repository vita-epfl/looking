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
parser.add_argument('--model', '-m', type=str, help='model type [resnet18, resnet50, alexnet]', default="resnet50")
parser.add_argument('--save', help='save the model', action='store_true')
parser.add_argument('--epochs', '-e', type=int, help='number of epochs for training', default=100)
parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate for training', default=0.0001)
parser.add_argument('--split', type=str, help='dataset split', default="video")
parser.add_argument('--kitti', help='evaluate on kitti', action='store_true')
parser.add_argument('--path', type=str, help='path for model saving', default='./models/')
parser.add_argument('--jaad_split_path', '-jsp', type=str, help='proportion for the training', default="JAAD_2k30/")
parser.add_argument('--split_path', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/splits/")
parser.add_argument('--data_path', '-dp', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/data/")


args = parser.parse_args()

EPOCHS = args.epochs
split = args.split
model_type = args.model
kitti = args.kitti

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


assert model_type in ['resnet18', 'resnet50']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
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
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)



jaad_train = JAAD_Dataset(DATA_PATH, JAAD_PATH,"train", SPLIT_PATH, split, data_transform)
jaad_val = JAAD_Dataset(DATA_PATH, JAAD_PATH, "val", SPLIT_PATH, split, data_transform)


if model_type=='resnet18':
    model = LookingNet_early_fusion_18(PATH_MODEL + "resnet18_head_{}_new_crops.p".format(split), PATH_MODEL + "looking_model_jaad_{}_full_kps.pkl".format(split), device).to(device)
elif model_type=='resnet50':
    model = LookingNet_early_fusion_50(PATH_MODEL + "resnet50_head_{}.pkl".format(split), PATH_MODEL + "looking_model_jaad_{}_full_kps.pkl".format(split), device).to(device)


dataset_loader = torch.utils.data.DataLoader(jaad_train, batch_size=64, shuffle=True)
dataset_loader_val = torch.utils.data.DataLoader(jaad_val, batch_size=32, shuffle=True)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(list(model.encoder_head.parameters()) + list(model.final.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.0005)

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
        torch.save(model.state_dict(), PATH_MODEL + "model_combined_{}_{}_new_crops.p".format(model_type, video))
    model.train()

data_test_jaad = JAAD_Dataset(DATA_PATH, JAAD_PATH,"test", SPLIT_PATH, split, data_transform)
ap, acc = data_test_jaad.evaluate(model, device, 10)
print('Performance on JAAD | acc:{} | ap:{}'.format(acc,ap))

if kitti:
    #model = []
    #model = torch.load(PATH_MODEL + "model_combined_{}_{}_new_crops.p".format(video, pose), map_location=torch.device(device))
    jaad_val = Kitti_Dataset(DATA_PATH, "test", pose)


    joints_test, labels_test = jaad_val.get_joints()

    out_test = model(joints_test.to(device))
    acc_test = binary_acc(out_test.to(device), labels_test.view(-1,1).to(device))
    ap_test = average_precision(out_test.to(device), labels_test.to(device))

    print("Kitti | AP : {} | Acc : {}".format(ap_test, acc_test))


    data_test= Kitti_Dataset(DATA_PATH, "test", pose)
    dataset_loader_test = torch.utils.data.DataLoader(data_test,batch_size=8, shuffle=True)
    model.eval()
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
