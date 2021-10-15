import os
import argparse
from glob import glob
import numpy as np
import json

file = open('../create_data/annotations.csv', 'r')

def parse_annotation(file):
    """
    Utility function to parse the LOOK annotation file to a dictionnary
    """
    dict_output = {}
    stats_data = {}
    for line in file:
        if line.split(',')[0] != 'filename':
            line = line[:-1]
            label = int(line.split(',')[-1])
            if label != -1:
                name_data = line.split(',')[0].split('/')[1]
                if name_data not in stats_data:
                    stats_data[name_data] = {'nb_instances':1, 'looking':label, 'nb_frames':[]}
                else:
                    stats_data[name_data]['looking'] += label
                    stats_data[name_data]['nb_instances'] += 1
                    stats_data[name_data]['nb_frames'].append(line.split(',')[0])
    for name in stats_data.keys():
        stats_data[name]['nb_frames'] = len(np.unique(stats_data[name]['nb_frames']))

    return dict_output, stats_data

_, stats = parse_annotation(file)
print(stats)