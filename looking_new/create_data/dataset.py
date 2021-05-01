from utils import JAAD_loader, JAAD_creator
import os
import argparse

parser = argparse.ArgumentParser(description='Parsing the JAAD dataset and creating the data')
parser.add_argument('--path', dest='p', type=str, help='path to JAAD repository', default="/home/younesbelkada/Travail/JAAD")
parser.add_argument('--path_joints', dest='pj', type=str, help='path to the joints', default="/home/younesbelkada/Travail/data/output_pifpaf_01_2k30")
parser.add_argument('--txt_out', dest='to', type=str, help='path to the output txt file', default="./splits_jaad")
parser.add_argument('--dir_out', dest='do', type=str, help='path to the output files (images + joints)', default="/home/younesbelkada/Travail/data/JAAD_2k30")
args = parser.parse_args()

jaad_path = args.p
path_joints = args.pj
txt_out = args.to
dir_out = args.do

jaad_loader = JAAD_loader(jaad_path)
dict_annotation = jaad_loader.generate()

jaad_creator = JAAD_creator(txt_out, dir_out, path_joints, os.path.join(jaad_path, 'images'))
jaad_creator.create(dict_annotation)