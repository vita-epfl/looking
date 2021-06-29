from utils import JAAD_loader, JAAD_creator, Kitti_creator
import os
import argparse

parser = argparse.ArgumentParser(description='Parsing the datasets and creating the data')
parser.add_argument('--dataset', dest='d', type=str, help='which dataset to create', default="JAAD")
parser.add_argument('--path', dest='p', type=str, help='path to JAAD repository', default="/home/younesbelkada/Travail/JAAD")
parser.add_argument('--path_joints', dest='pj', type=str, help='path to the joints', default="/home/younesbelkada/Travail/data/output_pifpaf_01_2k30")
parser.add_argument('--txt_out', dest='to', type=str, help='path to the output txt file', default="./splits_jaad")
parser.add_argument('--dir_out', dest='do', type=str, help='path to the output files (images + joints)', default="/home/younesbelkada/Travail/data/JAAD_2k30")
parser.add_argument('--path_kitti', dest='p_kt', type=str, help='path to Kitti images', default="/home/younesbelkada/Travail/data/Kitti_images")
parser.add_argument('--path_kitti_anno', dest='p_a_kt', type=str, help='path to Kitti annotation', default="/home/younesbelkada/Travail/data/kitti_new")


args = parser.parse_args()

jaad_path = args.p
path_joints = args.pj
txt_out = args.to
dir_out = args.do
path_kitti_images = args.p_kt
path_kitti_anno = args.p_a_kt
data = args.d

if 'JAAD' in data:
    jaad_loader = JAAD_loader(jaad_path, txt_out)
    dict_annotation = jaad_loader.generate()

    jaad_creator = JAAD_creator(txt_out, dir_out, path_joints, os.path.join(jaad_path, 'images'))
    jaad_creator.create(dict_annotation)

if 'Kitti' in data:
    path_train = os.path.join(path_kitti_images, 'Train')
    path_test = os.path.join(path_kitti_images, 'Test')

    path_train_anno = os.path.join(path_kitti_anno, 'Trainset_2k30')
    path_test_anno = os.path.join(path_kitti_anno, 'Testset_2k30')

    kitti_creator = Kitti_creator(path_train, path_train_anno, path_test, path_test_anno, dir_out, txt_out)
    kitti_creator.create()