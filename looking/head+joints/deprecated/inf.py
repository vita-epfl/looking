from utils import *
from dataset import *
from network import *
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

video = True

model = LookingNet_early_fusion_4("../head/models/resnet_head_video_best.p", "../joints/models/looking_model_jaad_video.p", device).to(device)

model.load_state_dict(torch.load("./models/model_early_stop_5_video_04.p"))
model.eval()

model.cuda()

data_test = JAAD_Dataset("test", video, data_transform)
#data_test = Kitti_Dataset("test", data_transform)

dataset_loader_test = torch.utils.data.DataLoader(data_test,batch_size=8, shuffle=True)

acc = 0
ap = 0
out_lab = torch.Tensor([]).type(torch.float).to(device)
test_lab = torch.Tensor([]).to(device)
for heads, keypoints, y_test in dataset_loader_test:
    if use_cuda:
        heads, keypoints, y_test = heads.cuda(), keypoints.cuda(), y_test.cuda()
    output = model(heads, keypoints)
    #out_pred, pred_label = torch.max(nn.Softmax(dim=-1)(output), dim=-1)
    #ut_pred, pred_label = torch.max(output, dim=-1)
    out_pred = output
    le = heads.shape[0]
    pred_label = torch.round(out_pred)
    #acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
    #ap += le*average_precision(out_pred, y_test)
    test_lab = torch.cat((test_lab.detach().cpu(), y_test.detach().cpu().view(-1)), dim=0)
    #out_lab = torch.cat((out_lab.detach().cpu(), nn.Sigmoid(out_pred.detach().cpu()).view(-1)), dim=0)
    out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)

ap = average_precision(out_lab, test_lab)
acc = binary_acc(torch.round(out_lab).type(torch.float).view(-1), test_lab).item()
print('epoch {} | acc:{} | ap:{}'.format(0, acc,ap))
