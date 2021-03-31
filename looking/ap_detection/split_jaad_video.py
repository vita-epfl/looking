import os
import numpy as np

np.random.seed(0)

# put gt file

file = open("ground_truth_2k30.txt", "r")
file_test = open("../splits/JAAD_test.txt", "r")


def extract_scenes(tab, file):
    for line in file:
        line = line[:-1]
        tab.append(line)
    return tab


train_videos = []
val_videos = []
test_videos = []

test_videos =  extract_scenes(test_videos, file_test)


file_test.close()

# final files

file_test = open("jaad_test_scenes_2k30_ap.txt", "w")

#

data = {"path": [], "names": [], "bb": [], "im": [], "label": [], "video": [], "iou": [], "sc":[]}

for line in file:
    line = line[:-1]
    line_s = line.split(",")
    video = line_s[0].split("/")[0]
    data['path'].append(line_s[0])
    data['names'].append(line_s[1])
    data["bb"].append([float(line_s[2]), float(line_s[3]), float(line_s[4]), float(line_s[5])])
    data["im"].append(line_s[-2])
    data["label"].append(int(line_s[-1]))
    data["video"].append(video)
    data["iou"].append(float(line_s[-4]))
    data["sc"].append(float(line_s[-3]))

for i in range(len(data["label"])):
    line = ','.join(
        [data["path"][i], data["names"][i], str(data["bb"][i][0]), str(data["bb"][i][1]), str(data["bb"][i][2]),
         str(data["bb"][i][3]), data["im"][i], str(data["sc"][i]), str(data['label'][i])])
    if data["video"][i] in test_videos: #and data['iou'][i] >= 0.5:
        file_test.write(line + '\n')

file_test.close()
