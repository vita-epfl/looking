import sys
sys.path.append('..')
from create_data.utils import *
import os
import argparse

parser = argparse.ArgumentParser(description='Parsing the JAAD dataset and computing the AP on the test set')
parser.add_argument('--path', dest='p', type=str, help='path to official Kitti annotations', default="/home/younesbelkada/Travail/data/drive-download-20210430T173538Z-001/label_2")
parser.add_argument('--path_pif', dest='p_pif', type=str, help='path to pifpafs annotation on the Kitti dataset', default="/home/younesbelkada/Travail/Project/Kitti/anno_2k30/kitti_2k30/test/")
parser.add_argument('--path_pred', dest='p_pred', type=str, help='path to our Kitti test set', default="/home/younesbelkada/Travail/data/Kitti/Kitti_images/Test")
parser.add_argument('--enlarge', dest='l', type=float, help='Enlarge param', default=1)

args = parser.parse_args()
kitti_path = args.p
pif_path = args.p_pif
test_path = args.p_pred
enlarge = args.l

kitti_loader = Kitti_label_loader(kitti_path, pif_path, test_path)
di_gt, di_predicted, scores = kitti_loader.create_dicts(enlarge)

ap_computer = AP_computer(di_gt, di_predicted, scores, 'Kitti')
ap_computer.create_txt()