import json
import os

path = 'labels_2d/'
#file_name = "nvidia-aud-2019-04-18_0_image0.json"

file_names= ["meyer-green-2019-03-16_0_image0.json",
"huang-lane-2019-02-12_0_image0.json",
"clark-center-2019-02-28_0_image0.json",
"bytes-cafe-2019-02-07_0_image4.json",
"bytes-cafe-2019-02-07_0_image8.json",
"nvidia-aud-2019-04-18_0_image0.json"]


tab = {}
for file_name in file_names:
	name = file_name.split('_')[0]
	data = json.load(open(path+file_name, 'r'))
	if name not in tab:
		tab[name] = []
	for key in data['labels']:
		for di in data["labels"][key]:
			if di["label_id"] not in tab[name]:
				tab[name].append(di["label_id"])
	#break

s = 0
for key in tab:
	s += len(tab[key])


print(s)
