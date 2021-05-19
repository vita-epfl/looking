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


def new_crop(img, kps):
	default_l = 10
	idx = [2, 3]

	kps1, kps2 = [kps[idx[0]], kps[17+idx[0]], kps[34+idx[0]]], [kps[idx[1]], kps[17+idx[1]], kps[34+idx[1]]]
	# l is euclidean dist between keypoints
	l = np.linalg.norm(np.array(kps2)-np.array(kps1))
	if l < default_l:
		l = default_l
	bbox = [kps1[0]-1*l, kps1[1]-2*l, kps2[0]+1*l, kps2[1]+2*l]
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])

	x1, y1, x2, y2 = bbox

	# No head in picture
	default_l = 5
	if x1 == img.shape[1]:
		x1 -= default_l
	if y1 == img.shape[0]:
		y1 -= default_l
	if x2 == 0:
		x2 += default_l
	if y2 == 0:
		y2 += default_l
	if x1 == x2:
		x2 += default_l
	if y1 == y2:
		y2 += default_l

	# shape img 1080*1920
	if y2 > img.shape[0]:
		y2 = img.shape[0]
	if x2 > img.shape[1]:
		x2 = img.shape[1]
	return img[int(y1):int(y2), int(x1):int(x2)]

train = True
test = True

if train:
	"""path_train_images = '../data/Kitti/Trainset/'
	path_train = '../data/Kitti/Trainset/anno_Trainset/'
	final_di = {'head':[], 'keypoints':[], 'Y':[], 'names':[]}
	path_out = "../data/Kitti/" """

	path_train_images = '../data/Kitti/Trainset/'
	path_train = '../data/Kitti/Trainset/anno_Trainset/'
	final_di = {'head':[], 'keypoints':[], 'Y':[], 'names':[]}
	path_out = "../data/Kitti_new_crops/"

	name_pic = 0
	file_out = open("kitti_gt_new_crops.txt", "w")
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
			head = new_crop(im, keypoints)
			file_out.write(name+','+name_head+',train,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')

			di = {"X":keypoints}
			json.dump(di, open(path_out+name_head+".json", 'w'))

			Image.fromarray(head).convert('RGB').save(path_out+name_head)
			name_pic += 1

	#json.dump(final_di, open("kitti_head_joints_train.json", 'w'))

if test:
	"""path_train_images = '../data/Kitti/Testset/'
	path_train = '../data/Kitti/Testset/anno_Testset/'"""

	path_train_images = '../data/Kitti/Testset/'
	path_train = '../data/Kitti/Testset/anno_Testset/'

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
			head = new_crop(im, keypoints)

			file_out.write(name+','+name_head+',test,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')
			di = {"X":keypoints}
			json.dump(di, open(path_out+name_head+".json", 'w'))

			Image.fromarray(head).convert('RGB').save(path_out+name_head)
			name_pic += 1



file_out.close()
