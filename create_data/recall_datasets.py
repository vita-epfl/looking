import os
import csv
import argparse
from utils import JAAD_loader, extract_scenes

parser = argparse.ArgumentParser(description='Parsing the datasets and creating the data')
parser.add_argument('--path_jaad', dest='p', type=str, help='path to JAAD repository', default="/data/taylor-data/Datasets/JAAD")
args = parser.parse_args()

path_jaad = args.p
JAAD_annotations = JAAD_loader(path_jaad, '')
annotation_jaad = JAAD_annotations.generate()

def parse_annotation(file):
    """
    Utility function to parse the LOOK annotation file to a dictionnary
    """
    dict_output = {}
    stats_data = {}
    
    for line in file:
        if line.split(',')[0] != 'filename':
            line = line[:-1]
            image_name = line.split(',')[0]
            if image_name not in dict_output:
                dict_output[image_name] = {'bboxes':[], 'splits':[], 'labels':[], 'lines':[]}
            bbox = [float(line.split(',')[1]), float(line.split(',')[2]), float(line.split(',')[3]), float(line.split(',')[4])]
            label = int(line.split(',')[-1])
            split = line.split(',')[5]
            if label != -1:
                dict_output[image_name]['bboxes'].append(bbox)
                dict_output[image_name]['splits'].append(split)
                dict_output[image_name]['labels'].append(label)
                name_data = line.split(',')[0].split('/')[1]
                if name_data not in stats_data:
                    stats_data[name_data] = {}
                    if split not in stats_data[name_data]:
                        stats_data[name_data][split] = 1
                else:
                    if split not in stats_data[name_data]:
                        stats_data[name_data][split] = 1
                    else:
                        stats_data[name_data][split] += 1
    return dict_output, stats_data

def compute_recall_jaad(annotations_jaad, array_test_set, matched_file_txt):
    """[summary]

    Args:
        annotations_jaad ([type]): [description]
        array_test_set ([type]): [description]
        matched_file_txt ([type]): [description]
    """
    test_instances = 0
    matched_instances = sum(1 for line in matched_file_txt)
    for i in range(len(annotations_jaad['Y'])):
        if annotation_jaad['video'][i] in array_test_set:
            test_instances += 1
    return matched_instances / test_instances

def compute_recall_look(annotations_look, matched_file_txt):
    matched_data = {'JRDB':0, 'Kitti':0, 'Nuscenes':0}
    for line in matched_file_txt:
        line = line[:-1]
        data_name = line.split(',')[1]
        split = line.split(',')[2]
        if split == 'test':
            matched_data[data_name] += 1
    
    print('JRDB: {} | Nuscenes: {} | Kitti: {}'.format((matched_data['JRDB']/annotations_look['JRDB']['test']), (matched_data['Nuscenes']/annotations_look['Nuscenes']['test']), (matched_data['Kitti']/annotations_look['Kitti']['test'])))

print(compute_recall_jaad(annotation_jaad, extract_scenes(open('./splits_jaad/test.txt', 'r')), open('./splits_jaad/jaad_test_scenes.txt', 'r')))

_, annotations_look = parse_annotation(open('annotations.csv', 'r'))
compute_recall_look(annotations_look, open('./splits_look/ground_truth_look.txt', 'r'))