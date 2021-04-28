import argparse
import os
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, help='dataset path')
parser.add_argument('--name', '-n', type=str, help='dataset name')
parser.add_argument('--train', '-tr', type=str, help='train file path')
parser.add_argument('--val', '-v', type=str, help='validation file path')
parser.add_argument('--test', '-ts', type=str, help='test file path')
args = parser.parse_args()

data_path = args.path
data_name = args.name
train_file = args.train
val_file = args.val
test_file = args.test


split_types = ['train', 'val', 'test']
split_files = [train_file, val_file, test_file]


if data_name == 'JAAD':
    images_path = data_path + 'images/'
    col_names = ['video_name', 'pedestrian_code', 'frame_number', 'bounding_box_y1', 'bounding_box_x1', 'bounding_box_y2', 'bounding_box_x2', 'looking_label']
    anno_df = pd.read_csv(data_path + 'results_new.csv', header=None, names=col_names)



elif data_name == 'PIE':
    images_path = 'PIE_clips/'
    col_names = ['set_name', 'video_name', 'pedestrian_code', 'frame_number', 'bounding_box_y1', 'bounding_box_x1', 'bounding_box_y2', 'bounding_box_x2', 'looking_label']
    anno_df = pd.read_csv(data_path + 'results_new.csv', header=None, names=col_names)


print(data_path[:-1])
print('-- Global stats --')
if data_name in ['JAAD', 'PIE']:
    unique_pedestrians = len(set(anno_df['pedestrian_code']))
    num_anno = len(anno_df)
    pos_instances, neg_instances = len(anno_df[anno_df['looking_label']==1]), len(anno_df[anno_df['looking_label']==0])

    if data_name == 'PIE':
        print(f"Number of frames containing pedestrians: {len(set(tuple(x) for x in anno_df[['video_name', 'frame_number']].values))}")
    else:
        print(f"Number of frames containing pedestrians: {len(set(tuple(x) for x in anno_df[['set_name', 'video_name', 'frame_number']].values))}")
    print(f'Unique pedestrians: {unique_pedestrians}')
    print(f'Total number of instances: {num_anno}')
    print(f'Positive instances: {pos_instances}   |   Negative instances: {neg_instances}')
    print(f'Average number of frames per pedestrian: {int(num_anno/unique_pedestrians)}')
    print()


    if len(split_files) > 0:
        print('-- Split stats --')

    for i, f_path in enumerate(split_files):
        print(split_types[i])
        file = open(f'{f_path}', 'r')
        split_videos = [line.split('\n')[0] for line in file.readlines()]
        if data_name == 'PIE':
            split_df = anno_df.loc[anno_df['set_name'].isin(split_videos)]
        elif data_name == 'JAAD':
            split_df = anno_df.loc[anno_df['video_name'].isin(split_videos)]

        unique_pedestrians = len(set(split_df['pedestrian_code']))
        num_anno = len(split_df)
        pos_instances, neg_instances = len(split_df[split_df['looking_label']==1]), len(split_df[split_df['looking_label']==0])

        print(f'Unique pedestrians: {unique_pedestrians}')
        print(f'Total number of instances: {num_anno}')
        print(f'Positive instances: {pos_instances}   |   Negative instances: {neg_instances}')
        print(f'Average number of frames per pedestrian: {int(num_anno/unique_pedestrians)}')
        print()

else:
    anno_folders = []
    anno_files = []
    for root, sub_dirs, files in os.walk(data_path):
        if 'anno' in root:
            anno_folders.append(root)
            all_files = os.listdir(root)
            json_files = [root + '/' + f for f in all_files if 'json' in f]
            anno_files.extend(json_files)


    labels = []
    for f_path in anno_files:
        with open(f_path) as f_json:
            f = json.load(f_json)
            labels.extend(f.get('Y'))

    print(f'Number of frames containing pedestrians: {len(anno_files)}')
    print(f'Unique pedestrians: TODO')
    print(f'Total number of instances: {len(labels)}')
    print(f'Positive instances: {len([l for l in labels if l == 1])}   |   Negative instances: {len([l for l in labels if l == 0])}')

    if len(split_files) > 0:
        print('-- Split stats --')

    for i, split_path in enumerate(split_files):
        if split_path:
            print(split_types[i])
            split_file = open(f'{split_path}', 'r')
            split_videos = [line.split('\n')[0] for line in split_file.readlines()]
            split_files = [f for f in anno_files if any(s in f for s in split_videos)]

            split_labels = []
            for f_path in split_files:
                with open(f_path) as f_json:
                    f = json.load(f_json)
                    split_labels.extend(f.get('Y'))

            print(f'Number of frames containing pedestrians: {len(split_files)}')
            print(f'Unique pedestrians: TODO')
            print(f'Total number of instances: {len(split_labels)}')
            print(f'Positive instances: {len([l for l in split_labels if l == 1])}   |   Negative instances: {len([l for l in split_labels if l == 0])}')
            print()
