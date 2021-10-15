import sys
sys.path.append('..')
from create_data.utils import *
import os
import argparse

parser = argparse.ArgumentParser(description='Parsing the JAAD dataset and computing the AP on the test set')
parser.add_argument('--path', dest='p', type=str, help='path to official JAAD repository', default="/home/younesbelkada/Travail/JAAD")
parser.add_argument('--txt_out', dest='to', type=str, help='path to the output txt file', default="../create_data/splits_jaad")


args = parser.parse_args()
jaad_path = args.p
txt_path = args.to

jaad_loader = JAAD_loader(jaad_path, txt_path)

ground_truth_bboxes = jaad_loader.generate_ap_gt_test()
predicted_bboxes, predicted_scores = jaad_loader.generate_ap_test(1)

ap_computer = AP_computer(ground_truth_bboxes, predicted_bboxes, predicted_scores)
ap_computer.create_txt()

