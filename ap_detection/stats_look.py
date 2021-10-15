import os
import argparse
from glob import glob
import numpy as np
import json

parser = argparse.ArgumentParser(description='Compute the stats on the LOOK dataset')
parser.add_argument('--path', dest='p', type=str, help='path to the LOOK annotations ', default="/data/younes-data/")
parser.add_argument('--txt_out', dest='to', type=str, help='path to the input txt file', default="../create_data/")
parser.add_argument('--name', dest='n', type=str, help='name of the dataset', default='kitti')

def get_train_test(path_txt, name_='jack'):
    test, train = [], []
    with open(os.path.join(path_txt, 'test.txt')) as file:
        for line in file:
            name = line[:-1]
            if name_ == 'nu':
                name = 'Nu_images/'+name
            elif name_ == 'kitti':
                name = 'Kitti_images/'+name
            test.append(name)
    with open(os.path.join(path_txt, 'trainval.txt')) as file:
        for line in file:
            name = line[:-1]
            if name_ == 'nu':
                name = 'Nu_images/'+name
            elif name_ == 'kitti':
                name = 'Kitti_images/'+name
            train.append(name)
    return train, test

def get_annotations(path_data, folder_name, folders_array):
    nb_look = 0
    nb_not_look = 0
    nb_dont_know = 0
    total_annotations = 0
    frames = 0
    for scenes in folders_array:
        path_annotation_data = os.path.join(path_data, folder_name, scenes, 'anno_'+scenes.split('/')[-1])
        frames += len(glob(path_annotation_data+'/*.json'))
        for annotations in glob(path_annotation_data+'/*.json'):
            di_annotation = json.load(open(annotations, 'r'))
            array_look = np.array(di_annotation['Y'])
            #print(array_look[array_look != -1])
            nb_not_look += len(array_look[array_look == 0])
            nb_look += len(array_look[array_look == 1])
            nb_dont_know += len(array_look[array_look == -1])
            total_annotations += len(array_look[array_look != -1])
            #exit(0)
        #print(path_annotation_data)
    return nb_look, nb_not_look, total_annotations, frames, nb_dont_know
args = parser.parse_args()

path_annotations = args.p
path_txt = args.to
name = args.n

if name=='jack':
    folder_name = 'JackRabbot' 
elif name == 'kitti':
    folder_name = 'Kitti'
elif name == 'nu':
    folder_name = 'Nuscenes'
path_txt += 'splits_{}'.format(name)

train_folder, test_folders = get_train_test(path_txt, name)
print(train_folder, test_folders)
nb_look, nb_not_look, total_annotations, frames, nb_dont_know = get_annotations(path_annotations, folder_name, train_folder)
print('Train set | look : {} | not look : {} | total annotated instances : {} | nb frames : {} | nb dont know : {}'.format(nb_look, nb_not_look, total_annotations, frames, nb_dont_know))

nb_look, nb_not_look, total_annotations, frames, nb_dont_know = get_annotations(path_annotations, folder_name, test_folders)
print('Test set | look : {} | not look : {} | total annotated instances : {} | nb frames : {} |  nb dont know : {}'.format(nb_look, nb_not_look, total_annotations, frames, nb_dont_know))