import os

path_images = os.listdir('/data/younes-data/JAAD_original/JAAD/images')
path_annotations = os.listdir('/data/younes-data/LOOK_annotations/JAAD')
print(len(path_images))
exit(0)
for folders in path_images:
    len_anno = len(os.listdir('/data/younes-data/LOOK_annotations/JAAD/'+folders))
    len_gt = len(os.listdir('/data/younes-data/JAAD_original/JAAD/images/'+folders))
    print(folders, len_anno, len_gt)
    if len_anno != len_gt:
        print(folders)