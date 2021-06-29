from utils import *
import os
import argparse

parser = argparse.ArgumentParser(description='Parsing the JAAD dataset and creating the data')
parser.add_argument('--gt_name', dest='gt', type=str, help='name of the gt file', default="ground_truth.txt")
parser.add_argument('--txt_out', dest='to', type=str, help='path to the output txt file', default="./splits_jaad")
args = parser.parse_args()

gt_file = args.gt
txt_out = args.to

jaad_splitter = JAAD_splitter(gt_file, txt_out)
jaad_splitter.split_(jaad_splitter.data, 'scenes')
jaad_splitter.split_(jaad_splitter.data, 'instances')