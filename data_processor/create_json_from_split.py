import json
import os


data_path = 'Nuscenes'
train_txt = data_path + '/train_nuscenes.txt'
test_txt = data_path + '/test_nuscenes.txt'
output = data_path + '/json_files/'

split_types = ['train', 'test']
split_files = [train_txt, test_txt]

anno_folders = []
anno_files = []
for root, sub_dirs, files in os.walk(data_path):
    if 'anno_' in root:
        anno_folders.append(root)
        all_files = os.listdir(root)
        json_files = [root + '/' + f for f in all_files if 'json' in f]
        anno_files.extend(json_files)

X = []
Y = []
bboxes = []
for f_path in anno_files:
    with open(f_path) as f_json:
        f = json.load(f_json)
        X.extend(f.get('X'))
        Y.extend(f.get('Y'))
        bboxes.extend(f.get('bbox'))

print('all', len(X), len(Y), len(bboxes))

all_out = {}
all_out['X'] = X
all_out['Y'] = Y
all_out['bboxes'] = bboxes

with open(output + data_path + '_all.json', 'w') as fp:
    json.dump(all_out, fp)


for i, split_path in enumerate(split_files):
    if split_path:
        X_split = []
        Y_split = []
        bboxes_split = []
        split_file = open(f'{split_path}', 'r')
        split_videos = [line.split('\n')[0] for line in split_file.readlines()]
        split_files = [f for f in anno_files if any(s in f for s in split_videos)]

        split_labels = []
        for f_path in split_files:
            with open(f_path) as f_json:
                f = json.load(f_json)
                X_split.extend(f.get('X'))
                Y_split.extend(f.get('Y'))
                bboxes_split.extend(f.get('bbox'))
        print(split_types[i], len(X_split), len(Y_split), len(bboxes_split))
        all_out_split = {}
        all_out_split['X'] = X_split
        all_out_split['Y'] = Y_split
        all_out_split['bboxes'] = bboxes_split

        with open(output + data_path + f'_{split_types[i]}.json', 'w') as fp:
            json.dump(all_out_split, fp)
