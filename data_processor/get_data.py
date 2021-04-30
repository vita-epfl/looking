import os
from shutil import copyfile

anno_txt = ''
out_path = ''
folder_to_search = '/work/vita/datasets/NUSCENES/Asia/samples/samples/'

if not os.path.exists(out_path):
    os.makedirs(out_path)

anno_file = open(f'{anno_txt}', 'r')
anno_files = [line.split('\n')[0].split('.')[:-1] for line in anno_file.readlines()]

files_to_copy = []

for folder in os.listdir(folder_to_search):
    folder_files = os.listdir(folder_to_search + folder)
    files_to_keep = [f for f in folder_files if f in anno_files]
    print(folder, len(files_to_keep))
    files_to_copy.extend(files_to_keep)

    for f_out in files_to_keep:
        copyfile(folder_to_search + folder + '/' + f_out, out_path + f_out)

print(len(files_to_copy), len(anno_files))
