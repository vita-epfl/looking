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

parser = argparse.ArgumentParser(description='Analysis of the head model on JAAD')

# parameters

parser.add_argument('--path', type=str, help='path for model saving', default='./models/')
parser.add_argument('--jaad_split_path', '-jsp', type=str, help='proportion for the training', default="JAAD_2k30/")
parser.add_argument('--split_path', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/splits/")
parser.add_argument('--data_path', '-dp', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/data/")
parser.add_argument('--pose',  type=str, help='pose type', default="full")
parser.add_argument('--split', type=str, help='dataset split', default="video")


args = parser.parse_args()

pose = args.pose
video = args.split

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


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)


model = torch.load(PATH_MODEL + "looking_model_jaad_{}_{}_kps_romain.p".format(video, pose), map_location=torch.device(device))
model.eval()
jaad_test = JAAD_Dataset_joints(DATA_PATH, JAAD_PATH, "test", SPLIT_PATH, pose, video)

ap, acc, f_pos, f_neg = jaad_test.get_mislabeled_test(model, device)

with open("false_pos_joints.txt", "w") as output:
    for row in f_pos:
        output.write(str(row)+'\n')
with open("false_neg_joints.txt", "w") as output:
    for row in f_neg:
        output.write(str(row)+'\n')

print("AP JAAD : {} | Acc JAAD : {}".format(ap, acc))
print(f"False positives: {len(f_pos)}, False negatives: {len(f_neg)}")
