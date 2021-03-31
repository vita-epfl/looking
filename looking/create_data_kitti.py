import PIL
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import json

def enlarge_bbox(bb, enlarge=1):
	delta_h = (bb[3]) / (7 * enlarge)
	delta_w = (bb[2]) / (3.5 * enlarge)
	bb[0] -= delta_w
	bb[1] -= delta_h
	bb[2] += delta_w
	bb[3] += delta_h
	return bb

def crop(img, bbox):
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])
	x1, y1, w, h = bbox
	h /= 3
	return img[int(y1):int(y1+h), x1:int(x1+w)]

train = True
test = True

if train:
	path_train_images = '/data/younes-data/Kitti/Train/'
	path_train = '/data/younes-data/Kitti/annotations/Trainset_2k30/'
	final_di = {'head':[], 'keypoints':[], 'Y':[], 'names':[]}
	path_out = "./data/Kitti/"
	name_pic = 0
	file_out = open("kitti_gt.txt", "w")
	#print(glob(path_train+'*.json'))
	for file in tqdm(glob(path_train+'*.json')):
		data = json.loads(open(file, 'r').read())
		name = file.split('/')[-1][:-5]
		pil_im = PIL.Image.open(path_train_images+name).convert('RGB')
		im = np.asarray(pil_im)
		for d in range(len(data["Y"])):
			name_head = str(name_pic).zfill(10)+'.png'
			keypoints = data["X"][d]
			
			label = data["Y"][d]
			
			bbox = enlarge_bbox(data['bbox'][d])
			head = crop(im, bbox)

			file_out.write(name+','+name_head+',train,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')

			di = {"X":keypoints}
			json.dump(di, open(path_out+name_head+".json", 'w'))

			Image.fromarray(head).convert('RGB').save(path_out+name_head)
			name_pic += 1
	
	#json.dump(final_di, open("kitti_head_joints_train.json", 'w'))

if test:
	path_train_images = '/data/younes-data/Kitti/Test/'
	path_train = '/data/younes-data/Kitti/annotations/Testset_2k30/'
	final_di = {'head':[], 'keypoints':[], 'Y':[], 'names':[]}


	for file in tqdm(glob(path_train+'*.json')):
		data = json.loads(open(file, 'r').read())
		name = file.split('/')[-1][:-5]
		pil_im = PIL.Image.open(path_train_images+name).convert('RGB')
		im = np.asarray(pil_im)
		for d in range(len(data["Y"])):
			name_head = str(name_pic).zfill(10)+'.png'
			keypoints = data["X"][d]
			
			label = data["Y"][d]
			
			bbox = enlarge_bbox(data['bbox'][d])
			head = crop(im, bbox)

			file_out.write(name+','+name_head+',test,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')

			di = {"X":keypoints}
			json.dump(di, open(path_out+name_head+".json", 'w'))

			Image.fromarray(head).convert('RGB').save(path_out+name_head)
			name_pic += 1
	


file_out.close()
