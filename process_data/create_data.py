import os
import json
from glob import glob
import sys
from PIL import Image
import numpy as np
import cv2

def parse_txt(path_txt, train_file, test_file):
	"""
	Utility function to parse the txt files and retrive the paths
	"""
	train = []
	test = []
	with open(path_txt+'/'+train_file, 'r') as file:
		for line in file:
			train.append(line[:-1])
	with open(path_txt+'/'+test_file, 'r') as file:
		for line in file:
			test.append(line[:-1])
	return train, test

def create_path(raw_path):
	"""
	A simple utility function to calculate the path to annotations
	"""
	raw_s = raw_path.split('/')
	name = raw_s[-1]
	return raw_path+'/anno_'+name

def convert_kps(data):
	X = []
	Y = []
	C = []
	A = []
	i = 0
	while i < 51:
		X.append(data[i])
		Y.append(data[i+1])
		C.append(data[i+2])
		i += 3
	A = np.array([X, Y, C]).flatten().tolist()
	return A

def enlarge_bbox(bb, enlarge=1):
	delta_h = (bb[3]) / (7 * enlarge)
	delta_w = (bb[2]) / (3.5 * enlarge)
	bb[0] -= delta_w
	bb[1] -= delta_h
	bb[2] += delta_w
	bb[3] += delta_h
	return bb

def convert_bb(bb):
	return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]


def crop_eyes(img, kps):
	default_l = 10
	idx = [1,2]
	#kps1, kps2 = kps[3*idx[0]:3*idx[0]+2], kps[3*idx[1]:3*idx[1]+2]
	candidate1, candidate2 = [kps[idx[0]], kps[17+idx[0]], kps[34+idx[0]]], [kps[idx[1]], kps[17+idx[1]], kps[34+idx[1]]]
	if candidate1[0] > candidate2[0]:
		kps1 = candidate2
		kps2 = candidate1
	else:
		kps1 = candidate1
		kps2 = candidate2
	#img = im.copy()
	try:
		#print(img[[int(k) for k in kps1][0]][[int(k) for k in kps1][1]])
		#img[[int(k) for k in kps1][0]][[int(k) for k in kps1][1]] = [0,0,255]
		#img[[int(k) for k in kps2][0]][[int(k) for k in kps2][1]] = [0,0,255]
		#img = cv2.circle(img, ([int(k) for k in kps1][0],[int(k) for k in kps1][1]), radius=1, color=(0, 0, 255), thickness=-1)
		#img = cv2.circle(img, ([int(k) for k in kps2][0],[int(k) for k in kps2][1]), radius=1, color=(255, 0, 0), thickness=-1)
		# l is euclidean dist between keypoints
		l = np.linalg.norm(np.array(kps2)-np.array(kps1))
		center_x = (kps1[0] + kps2[0])/2
		center_y = (kps1[1] + kps2[1])/2
		"""if l < default_l:
			l = default_l"""
		# Not balanced as pifpad tends to locate eyes lower than they are
		bbox = [kps1[0]-0.9*max(l, default_l), kps1[1]-1.1*max(l, default_l), kps2[0]+0.9*max(l, default_l), kps2[1]+0.3*max(l, default_l)]
		#bbox = [kps1[0]-1.2*max(l, 5), kps1[1]-0.8*max(l, 5), kps2[0]+1.2*max(l, 5), kps2[1]+0.8*max(l, 5)]
		for i in range(len(bbox)):
			if bbox[i] < 0:
				bbox[i] = 0
			else:
				bbox[i] = int(bbox[i])

		x1, y1, x2, y2 = bbox

		# No head in picture
		default_l = 5
		if x1 >= img.shape[1]:
			x1 = img.shape[1]-default_l
		if y1 >= img.shape[0]:
			y1 = img.shape[0]-default_l
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
	except:
		return img[0:5, 0:5]

def crop_head(img, kps):
	default_l = 10
	idx = [3, 4]
	#kps1, kps2 = kps[3*idx[0]:3*idx[0]+2], kps[3*idx[1]:3*idx[1]+2]
	candidate1, candidate2 = [kps[idx[0]], kps[17+idx[0]], kps[34+idx[0]]], [kps[idx[1]], kps[17+idx[1]], kps[34+idx[1]]]
	if candidate1[0] > candidate2[0]:
		kps1 = candidate2
		kps2 = candidate1
	else:
		kps1 = candidate1
		kps2 = candidate2
	# l is euclidean dist between keypoints
	l = np.linalg.norm(np.array(kps2)-np.array(kps1))
	if l < default_l:
		l = default_l

	bbox = [kps1[0]-0.5*max(l, default_l), kps1[1]-1*max(l, default_l), kps2[0]+0.5*max(l, default_l), kps2[1]+1*max(l, default_l)]
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

def main(argv):
	path_txt = ''
	train, test = parse_txt(path_txt, argv[1], argv[2])
	j = 0
	if 'nu' in argv[3].lower():
		out = './nu/'
		out_eyes = './nu_experiment/eyes/'
		out_head = './nu_experiment/heads/'
		out_joints = './nu_experiment/joints/'
		out_full = './nu_experiment/full_images/'
	elif 'jack'in argv[3].lower():
		out = './jack/'
	else:
		print('please use a valid dataset among [nu, jack]')


	if 'nu' in out:
		file_out = open('gt_nu_new_test.txt', 'w')

	elif 'jack' in out:
		file_out = open('gt_jack.txt', 'w')

	for raw in test:
		path = create_path(raw)
		#print(raw)
		for anno in glob(path+'/*.json'):
			name_im = raw+'/'+anno.split('/')[-1][:-5]

			file = open(anno, 'r')
			json_di = json.load(file)
			for i in range(len(json_di["X"])):
				im = Image.open(name_im)
				out_name = str(j).zfill(10)+'.json'

				#di = convert_kps(json_di["X"][i])
				di = json_di["X"][i]

				bb = convert_bb(enlarge_bbox(json_di["bbox"][i], enlarge=1))

				eyes = crop_eyes(np.asarray(im), di)
				head = crop_head(np.asarray(im), di)

				im = np.asarray(im)
				im_arr_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
				cv2.rectangle(im_arr_bgr, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), thickness=1)
				im = cv2.cvtColor(im_arr_bgr, cv2.COLOR_BGR2RGB)

				Image.fromarray(im).convert('RGB').save(out_full+out_name+'.png')
				Image.fromarray(eyes).convert('RGB').save(out_eyes+out_name+'.png')
				Image.fromarray(head).convert('RGB').save(out_head+out_name+'.png')

				json.dump(di, open(out_joints+out_name, "w"))
				line = ','.join([anno[:-5], 'test', out_name, str(json_di["bbox"][i][0]), str(json_di["bbox"][i][1]), str(json_di["bbox"][i][0]+json_di["bbox"][i][2]),str(json_di["bbox"][i][1]+json_di["bbox"][i][-1]), str(json_di["Y"][i])+'\n'])
				file_out.write(line)
				j += 1
			file.close()
	file_out.close()




def __main__():
	if len(sys.argv) < 4:
		print("Usage create_data.py trainFile testFile dataset")
	else:
		main(sys.argv)

if __main__:
	__main__()
