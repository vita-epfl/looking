import os, errno
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='Utility function to run pifpaf on folders')
parser.add_argument('--path_data', dest='pd', type=str, help='path to the images', default="/data/younes-data/LOOK/LOOK")
parser.add_argument('--path_out', dest='po', type=str, help='path to the output files', default="/data/younes-data/LOOK_keypoints")
parser.add_argument('--check', dest='ch', type=str, help='pifpaf checkpoint', default="shufflenetv2k30")
parser.add_argument('--instance-threshold', dest='th', type=float, help='instance threshold', default=0.1)


args = parser.parse_args()

path_data = args.pd
path_out = args.po
instance_thresh = args.th
checkpoint = args.ch

subdirectories = [x[0] for x in os.walk(path_data) if len(glob(os.path.join(x[0], '*.png'))+glob(os.path.join(x[0], '*.jpg'))) != 0]


new_subdirectories = []
for s in subdirectories:
    new_string = s.replace(path_data, '')
    new_subdirectories.append(new_string[1:])

for sub_folders in new_subdirectories:
    directory_out = os.path.join(path_out, sub_folders)
    try:
        os.makedirs(directory_out)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if len(glob(os.path.join(os.path.join(path_data, sub_folders), '*.png'))) == 0:
        ext = '*.jpg'
    else:
        ext = '*.png'
    directory_in = os.path.join(path_data, sub_folders, ext)
    
    command = "python3 -m openpifpaf.predict --glob {} --json-output {} --force-complete-pose --checkpoint {} --instance-threshold {}".format(directory_in, directory_out, checkpoint, instance_thresh)
    os.system(command)
