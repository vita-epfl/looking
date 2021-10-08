import sys
sys.path.append('..')
from create_data.utils import *
import os
import argparse

parser = argparse.ArgumentParser(description='Parsing the LOOK dataset and computing the AP on the test set')
parser.add_argument('--path', dest='p', type=str, help='path to official Kitti/Jack∕Nu annotations', default="/home/younesbelkada/Travail/data/drive-download-20210430T173538Z-001/label_2")
parser.add_argument('--path_pif', dest='p_pif', type=str, help='path to pifpafs annotation on the LOOK dataset', default="/home/younesbelkada/Travail/Project/Kitti/anno_2k30/kitti_2k30/test/")
parser.add_argument('--path_pred', dest='p_pred', type=str, help='path to our LOOK test set', default="/home/younesbelkada/Travail/data/Kitti/Kitti_images/Test")
parser.add_argument('--enlarge', dest='l', type=float, help='Enlarge param', default=1)
parser.add_argument('--name', dest='n', type=str, help='name of the dataset', default='kitti')
parser.add_argument('--path_out_txt', dest='p_txt', type=str, help='output path for gt', default="./ground_truth")
parser.add_argument('--path_pred_txt', dest='p_pred_txt', type=str, help='output path for predicted people txt files', default="./detection_results")


args = parser.parse_args()
gt_path = args.p
pif_path = args.p_pif
test_path = args.p_pred
enlarge = args.l
name = args.n
path_out_gt_txt = args.p_txt
path_out_pred_txt = args.p_pred_txt

if name.lower() == 'kitti':
    kitti_loader = Kitti_label_loader(gt_path, pif_path, test_path)
    di_gt, di_predicted, scores = kitti_loader.create_dicts(enlarge)
    ap_computer = AP_computer(di_gt, di_predicted, scores, 'Kitti')
    ap_computer.create_txt()
elif name.lower() == 'jack':
    JR_label_loader = JR_label_loader(gt_path, pif_path, path_out_gt_txt, path_out_pred_txt, test_path)
    JR_label_loader.get_gt_txt_files()
    JR_label_loader.get_predicted_text_files()
elif name.lower() == 'nu':
    NU_label_loader = NU_label_loader(gt_path, pif_path, path_out_pred_txt, path_out_gt_txt)
    NU_label_loader.get_gt_annotations()
    NU_label_loader.get_predicted_text_files()
