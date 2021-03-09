import os
import numpy as np

np.random.seed(0)

# put gt file

file = open("../ground_truth_2k30.txt", "r")
file_train = open("JAAD_train.txt", "r")
file_val = open("JAAD_val.txt", "r")
file_test = open("JAAD_test.txt", "r")


def extract_scenes(tab, file):
    for line in file:
        line = line[:-1]
        tab.append(line)
    return tab


train_videos = []
val_videos = []
test_videos = []

train_videos, val_videos, test_videos = extract_scenes(train_videos, file_train), extract_scenes(val_videos,
                                                                                                 file_val), extract_scenes(
    test_videos, file_test)

file_train.close()
file_val.close()
file_test.close()

# final files

file_train = open("jaad_train_scenes_2k30.txt", "w")
file_val = open("jaad_val_scenes_2k30.txt", "w")
file_test = open("jaad_test_scenes_2k30.txt", "w")

#

data = {"path": [], "names": [], "bb": [], "im": [], "label": [], "video": [], "iou": []}

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
    data["iou"].append(float(line_s[-3]))

for i in range(len(data["label"])):
    line = ','.join(
        [data["path"][i], data["names"][i], str(data["bb"][i][0]), str(data["bb"][i][1]), str(data["bb"][i][2]),
         str(data["bb"][i][3]), data["im"][i], str(data['label'][i])])
    if data["video"][i] in train_videos:
        file_train.write(line + '\n')
    elif data["video"][i] in test_videos and data['iou'][i] >= 0.5:
        file_test.write(line + '\n')
    elif data["video"][i] in val_videos:
        file_val.write(line + '\n')

file_train.close()
file_val.close()
file_test.close()

"""
data = {"path":[], "names":[], "bb":[], "im":[], "label":[], "video":[]}

for line in file:
	line = line[:-1]
	line_s = line.split(",")
	video = line_s[0].split("/")[0]
	data['path'].append(line_s[0])
	data['names'].append(line_s[1])
	data["bb"].append([float(line_s[2]), float(line_s[3]), float(line_s[4]), float(line_s[5])])
	data["im"].append(line_s[6])
	data["label"].append(int(line_s[-1]))
	data["video"].append(video)

Y = np.array(data["label"])
idx_pos = np.where(Y == 1)[0]
N_pos = len(idx_pos)
idx_neg = np.where(Y==0)[0]
np.random.shuffle(idx_neg)
idx_neg = idx_neg[:N_pos]
idx = np.concatenate((idx_neg, idx_pos))


data['path'] = np.array(data["path"])[idx]
data['names'] = np.array(data["names"])[idx]
data['bb'] = np.array(data["bb"])[idx, :]
data['im'] = np.array(data["im"])[idx]
data['label'] = np.array(data["label"])[idx]
data["video"] = np.array(data["video"])[idx]

#print(len(data["label"]))


idx_or = np.array(range(len(data['label'])))
N = len(idx_or)
np.random.shuffle(idx_or)

for j in range(int(0.6*N)):
	i = idx_or[j]
	line = ','.join([data["path"][i], data["names"][i], str(data["bb"][i][0]), str(data["bb"][i][1]), str(data["bb"][i][2]), str(data["bb"][i][3]), data["im"][i], str(data['label'][i])])
	file_train.write(line+'\n')
for j in range(int(0.6*N), int(0.7*N)):
	i = idx_or[j]
	line = ','.join([data["path"][i], data["names"][i], str(data["bb"][i][0]), str(data["bb"][i][1]), str(data["bb"][i][2]), str(data["bb"][i][3]), data["im"][i], str(data['label'][i])])
	file_val.write(line+'\n')
for j in range(int(0.7*N), N):
	i = idx_or[j]
	line = ','.join([data["path"][i], data["names"][i], str(data["bb"][i][0]), str(data["bb"][i][1]), str(data["bb"][i][2]), str(data["bb"][i][3]), data["im"][i], str(data['label'][i])])
	file_test.write(line+'\n')



names_unique = np.unique(data["video"])


train = names_unique[:int(0.6*len(names_unique))]
test = names_unique[int(0.6*len(names_unique)):int(0.9*len(names_unique))]
val = names_unique[int(0.9*len(names_unique)):]

for i in range(len(data["label"])):
	line = ','.join([data["path"][i], data["names"][i], str(data["bb"][i][0]), str(data["bb"][i][2]), str(data["bb"][i][2]), str(data["bb"][i][3]), data["im"][i], str(data['label'][i])])
	if data["video"][i] in train:
		file_train.write(line+'\n')
	elif data["video"][i] in test:
		file_test.write(line+'\n')
	elif data["video"][i] in val:
		file_val.write(line+'\n')


file_train.close()
file_test.close()
file_val.close()

file.close()
"""
