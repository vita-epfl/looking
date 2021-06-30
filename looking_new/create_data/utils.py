
from PIL import Image
import os
import pickle
import numpy as np
from jaad_data import JAAD
import json
from tqdm import tqdm
import errno
from glob import glob
from PIL import Image, ImageFile
import numpy as np

np.random.seed(0)

ImageFile.LOAD_TRUNCATED_IMAGES = True

## Helper functions

def extract_scenes(file):
	tab = []
	for line in file:
		line = line[:-1]
		tab.append(line)
	return tab

def convert_file_to_data(file):
	data = {"path":[], "bbox":[], "names":[], "Y":[], "video":[]}
	for line in file:
		line = line[:-1]
		line_s = line.split(",")
		path = line_s[0]+'/'+line_s[2].zfill(5)+'.png'
		Y = int(line_s[-1])
		bbox = [float(line_s[3]), float(line_s[4]), float(line_s[5]), float(line_s[6])]
		data['path'].append(path)
		data["bbox"].append(bbox)
		data['Y'].append(Y)
		data['names'].append(line_s[1])
		data["video"].append(line_s[0])
	return data

def enlarge_bbox(bb, enlarge=1):
	delta_h = (bb[3]) / (7 * enlarge)
	delta_w = (bb[2]) / (3.5 * enlarge)
	bb[0] -= delta_w
	bb[1] -= delta_h
	bb[2] += delta_w
	bb[3] += delta_h
	return bb

def enlarge_bbox_kitti(bb, enlarge=1):
	delta_h = (bb[3]) / (7.5 * enlarge)
	delta_w = (bb[2]) / (3.5 * enlarge)
	bb[0] -= delta_w
	bb[1] -= delta_h
	bb[2] += delta_w
	bb[3] += delta_h
	return bb


def convert_bb(bb):
	return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]

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


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	if interArea == 0:
		return 0
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
	boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def crop_image(img, bbox):
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		if bbox[0] > img.shape[1]:
			bbox[0] = img.shape[1]
		if bbox[2] > img.shape[1]:
			bbox[2] = img.shape[1]
		if bbox[1] > img.shape[0]:
			bbox[1] = img.shape[0]
		if bbox[1] > img.shape[0]:
			bbox[1] = img.shape[0]
		else:
			bbox[i] = int(bbox[i])
	im_cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
	return im_cropped


def crop_head(img, kps, dataset):
	default_size = (20, 20)
	# ears
	idx = [3,4]
	if dataset == 'jaad':
		nose = [kps[0], kps[1]]
		candidate1, candidate2 = kps[3*idx[0]:3*idx[0]+2], kps[3*idx[1]:3*idx[1]+2]
	elif dataset == 'kitti':
		nose = [kps[0], kps[17]]
		candidate1, candidate2 = [kps[idx[0]], kps[17+idx[0]]], [kps[idx[1]], kps[17+idx[1]],]

	if candidate1[0] > candidate2[0]:
		kps1 = candidate2
		kps2 = candidate1
	else:
		kps1 = candidate1
		kps2 = candidate2

	# distance between ears
	l_ears = np.linalg.norm(np.array(kps2)-np.array(kps1))
	# distance between nose and ear
	l_nose1 = np.linalg.norm(np.array(kps1)-np.array(nose))
	l_nose2 = np.linalg.norm(np.array(kps2)-np.array(nose))
	l_x = max(l_nose1, l_nose2, l_ears)

	# center on the y axis
	if l_x == l_ears:
		center_x = (kps1[0] + kps2[0])/2
	elif l_x == l_nose1:
		center_x = (kps1[0] + nose[0])/2
	elif l_x == l_nose2:
		center_x = (kps2[0] + nose[0])/2

	bbox = [center_x-0.7*l_x, nose[1]-0.7*l_x, center_x+0.7*l_x, nose[1]+0.7*l_x]
	x1, y1, x2, y2 = bbox
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	# Head not in image, return black box of default size
	if max(x1, x2) <= 0 or min(x1, x2) >= img.shape[1] or max(y1, y2) <= 0 or min(y1, y2) >= img.shape[0]:
		not_in_pic = img.copy()
		not_in_pic.fill(0)
		return not_in_pic[0:default_size[1], 0:default_size[0]]

	# if only one is out of the frame
	if x2 > img.shape[1]:
		x2 = img.shape[1]
	if y2 > img.shape[0]:
		y1 = img.shape[0]
	if x1 < 0:
		x1 = 0
	if y1 < 0:
		y1 = 0

	# Person too far and everything is in the same pixel, return just the pixel
	if x1 == x2:
		x2 += 1
	if y1 == y2:
		y2 += 1

	return img[int(y1):int(y2), int(x1):int(x2)]

def crop_eyes(img, kps, dataset):
	default_size = (15, 10)
	# eyes
	idx = [1,2]

	if dataset == 'jaad':
		nose = [kps[0], kps[1], kps[2]]
		candidate1, candidate2 = kps[3*idx[0]:3*idx[0]+2], kps[3*idx[1]:3*idx[1]+2]
	elif dataset == 'kitti':
		nose = [kps[0], kps[17], kps[34]]
		candidate1, candidate2 = [kps[idx[0]], kps[17+idx[0]], kps[34+idx[0]]], [kps[idx[1]], kps[17+idx[1]], kps[34+idx[1]]]

	if candidate1[0] > candidate2[0]:
		kps1 = candidate2
		kps2 = candidate1
	else:
		kps1 = candidate1
		kps2 = candidate2

	# distance between eyes in x axis
	l = np.linalg.norm(np.array(kps2)-np.array(kps1))
	# center of the eyes on the y axis
	center_y = (kps1[1] + kps2[1])/2
	#l_y = np.linalg.norm(np.array(nose[1])-np.array(min(kps1[1], kps2[1])))
	bbox = [kps1[0]-0.3*l, center_y-0.5*l, kps2[0]+0.3*l, center_y+0.5*l]
	x1, y1, x2, y2 = bbox
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	# Eyes not in image, return black box of default size
	if max(x1, x2) <= 0 or min(x1, x2) >= img.shape[1] or max(y1, y2) <= 0 or min(y1, y2) >= img.shape[0]:
		not_in_pic = img.copy()
		not_in_pic.fill(0)
		return not_in_pic[0:default_size[1], 0:default_size[0]]

	# if only one is out of the frame
	if x2 > img.shape[1]:
		x2 = img.shape[1]
	if y2 > img.shape[0]:
		y1 = img.shape[0]
	if x1 < 0:
		x1 = 0
	if y1 < 0:
		y1 = 0

	# Person too far and everything is in the same pixel, return just the pixel
	if x1 == x2:
		x2 += 1
	if y1 == y2:
		y2 += 1

	return img[int(y1):int(y2), int(x1):int(x2)]


def file_to_dict(file):
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
	return data


####

class JAAD_loader():
	"""
		Class definition for JAAD annotation processing
	"""
	def __init__(self, path_jaad, folder_txt):
		self.imdb = JAAD(data_path=path_jaad)
		self.imdb.generate_database()
		self.data = pickle.load(open(os.path.join(path_jaad,'data_cache/jaad_database.pkl'), 'rb'))
		self.folder_txt = folder_txt

	def generate(self):
		"""
			Generate the annotations in a dictionary
		"""
		data = {"path":[], "bbox":[], "names":[], "Y":[], "video":[]}
		for videos in self.data.keys():
			ped_anno = self.data[videos]["ped_annotations"]
			for d in ped_anno:
				if 'look' in ped_anno[d]['behavior'].keys():
					for i in range(len(ped_anno[d]['frames'])):

						path = os.path.join(videos,str(ped_anno[d]['frames'][i]).zfill(5)+'.png')
						bbox = [ped_anno[d]['bbox'][i][0], ped_anno[d]['bbox'][i][1], ped_anno[d]['bbox'][i][2], ped_anno[d]['bbox'][i][3]]

						data['path'].append(path)
						data["bbox"].append(bbox)
						data['Y'].append(ped_anno[d]['behavior']['look'][i])
						data['names'].append(d)
						data["video"].append(videos)
		return data

	def generate_ap_gt_test(self):
		di_gt = {}

		file_test = open(os.path.join(self.folder_txt, 'test.txt'), 'r')
		test_videos = extract_scenes(file_test)
		for videos in self.data.keys():
			if videos in test_videos:
				ped_anno = self.data[videos]["ped_annotations"]
				for d in ped_anno:
					if 'look' in ped_anno[d]['behavior'].keys():
						for i in range(len(ped_anno[d]['frames'])):
							name = os.path.join(videos,str(ped_anno[d]['frames'][i]).zfill(5)+'.png')
							bbox = [ped_anno[d]['bbox'][i][0], ped_anno[d]['bbox'][i][1], ped_anno[d]['bbox'][i][2], ped_anno[d]['bbox'][i][3]]
							if name not in di_gt:
								di_gt[name] = []
							di_gt[name].append(bbox)
		return di_gt

	def generate_ap_test(self, enlarge=3):
		if not os.path.isfile(os.path.join(self.folder_txt, 'ground_truth.txt')):
			print('ERROR: Ground truth file not found, please create the JAAD dataset first')
			exit(0)

		file_gt = open(os.path.join(self.folder_txt, 'ground_truth.txt'), 'r')
		file_test = open(os.path.join(self.folder_txt, 'test.txt'), 'r')
		data = {"path": [], "names": [], "bb": [], "im": [], "label": [], "video": [], "iou": [], "sc":[]}

		test_videos =  extract_scenes(file_test)
		di = {}
		scores = {}
		for line in file_gt:
			line = line[:-1]
			line_s = line.split(",")
			video = line_s[0].split("/")[0]
			if video in test_videos:
				path = line_s[0]
				if path not in di:
					di[path] = []
					scores[path] = []

				bbox = [float(line_s[2]), float(line_s[3]), float(line_s[4]), float(line_s[5])]
				score = float(line_s[-3])

				#bbox = enlarge_bbox(bbox, enlarge)
				di[path].append(bbox)
				scores[path].append(score)
		return di, scores

class JAAD_creator():
	"""
		Class definition for creating our custom JAAD dataset for heads / joints / heads+joints training
	"""
	def __init__(self, txt_out, dir_out, path_joints, path_jaad):
		self.txt_out = txt_out
		self.dir_out = dir_out
		self.path_joints = path_joints
		self.path_jaad = path_jaad

		if not os.path.exists(os.path.join(self.dir_out, 'fullbodies')):
			os.makedirs(os.path.join(self.dir_out, 'fullbodies'))

		if not os.path.exists(os.path.join(self.dir_out, 'heads')):
			os.makedirs(os.path.join(self.dir_out, 'heads'))

		if not os.path.exists(os.path.join(self.dir_out, 'eyes')):
			os.makedirs(os.path.join(self.dir_out, 'eyes'))


	def create(self, dict_annotation):
		"""
			Create the files given the dictionary with the annotations
		"""

		file_out = open(os.path.join(self.txt_out, "ground_truth.txt"), "w+")
		name_pic = 0
		for i in tqdm(range(len(dict_annotation["Y"]))):
			path = dict_annotation["path"][i]
			data_json = json.load(open(os.path.join(self.path_joints,path+'.predictions.json'), 'r'))

			if len(data_json)> 0:
				iou_max = 0
				j = None
				bb_gt = dict_annotation["bbox"][i]
				sc = 0
				for k in range(len(data_json)):
					di =data_json[k]
					bb = convert_bb(enlarge_bbox(di["bbox"]))
					iou = bb_intersection_over_union(bb, bb_gt)
					if iou > iou_max:
						iou_max = iou
						j = k
						bb_final = bb
						sc = di["score"]

				if iou_max > 0.3:
					kps = convert_kps(data_json[j]["keypoints"])
					name_head = str(name_pic).zfill(10)+'.png'

					img = Image.open(os.path.join(self.path_jaad,path)).convert('RGB')
					img = np.asarray(img)

					fullbody = crop_image(img, bb_final)
					head = crop_head(img, kps, 'jaad')
					eyes = crop_eyes(img, kps, 'jaad')

					Image.fromarray(fullbody).convert('RGB').save(os.path.join(self.dir_out,'fullbodies/' + name_head))
					Image.fromarray(head).convert('RGB').save(os.path.join(self.dir_out,'heads/' + name_head))
					Image.fromarray(eyes).convert('RGB').save(os.path.join(self.dir_out,'eyes/' + name_head))

					kps_out = {"X":kps}

					json.dump(kps_out, open(os.path.join(self.dir_out,name_head+'.json'), "w"))
					line = path+','+dict_annotation["names"][i]+','+str(bb_final[0])+','+str(bb_final[1])+','+str(bb_final[2])+','+str(bb_final[3])+','+str(sc)+','+str(iou_max)+','+name_head+','+str(dict_annotation["Y"][i])+'\n'
					name_pic += 1
					file_out.write(line)
		file_out.close()

class JAAD_splitter():
	"""
		Class definition to create JAAD splits
	"""
	def __init__(self, file_gt, folder_txt='./splits_jaad'):
		self.folder_txt = folder_txt
		self.file_gt = open(os.path.join(folder_txt, file_gt), 'r')

		file_train = open(os.path.join(self.folder_txt, "train.txt"), "r")
		file_val = open(os.path.join(self.folder_txt, "val.txt"), "r")
		file_test = open(os.path.join(self.folder_txt, "test.txt"), "r")
		self.train_videos, self.val_videos, self.test_videos = extract_scenes(file_train), extract_scenes(file_val), extract_scenes(file_test)

		file_train.close()
		file_val.close()
		file_test.close()

		self.data = file_to_dict(self.file_gt)

	def split_(self, data, split='scenes'):
		file_train = open(os.path.join(self.folder_txt, "jaad_train_{}.txt".format(split)), "w")
		file_val = open(os.path.join(self.folder_txt, "jaad_val_{}.txt".format(split)), "w")
		file_test = open(os.path.join(self.folder_txt, "jaad_test_{}.txt".format(split)), "w")

		if split == 'scenes':
			for i in range(len(data["label"])):
				line = ','.join(
					[data["path"][i], data["names"][i], str(data["bb"][i][0]), str(data["bb"][i][1]), str(data["bb"][i][2]),
					str(data["bb"][i][3]), data["im"][i], str(data['label'][i])])
				if data["video"][i] in self.train_videos:
					file_train.write(line + '\n')
				elif data["video"][i] in self.test_videos and data['iou'][i] >= 0.5:
					file_test.write(line + '\n')
				elif data["video"][i] in self.val_videos:
					file_val.write(line + '\n')
		else:
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
			data["iou"] = np.array(data["iou"])[idx]

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
				if data['iou'][i] >= 0.5:
					file_test.write(line+'\n')

		file_train.close()
		file_val.close()
		file_test.close()

class Kitti_creator():
	"""
		Wrapper class for Kitti dataset creation
	"""
	def __init__(self, path_images_train, path_annotation_train, path_images_test, path_annotation_test, path_out, path_out_txt):
		self.path_images_train = path_images_train
		self.path_annotation_train = path_annotation_train
		self.path_images_test = path_images_test
		self.path_annotation_test = path_annotation_test
		self.path_out = path_out
		self.path_out_txt = path_out_txt

		if not os.path.exists(os.path.join(self.path_out, 'fullbodies')):
			os.makedirs(os.path.join(self.path_out, 'fullbodies'))

		if not os.path.exists(os.path.join(self.path_out, 'heads')):
			os.makedirs(os.path.join(self.path_out, 'heads'))

		if not os.path.exists(os.path.join(self.path_out, 'eyes')):
			os.makedirs(os.path.join(self.path_out, 'eyes'))


		try:
			os.makedirs(self.path_out_txt)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	def create(self):
		"""
			Create the Kitti dataset
		"""
		print('Creating the Kitti dataset...')
		name_pic = 0
		file_out = open(os.path.join(self.path_out_txt,"ground_truth_kitti.txt"), "w")
		"""for idx_line, file in enumerate(tqdm(glob(os.path.join(self.path_annotation_train,'*.json')))):
			data = json.loads(open(file, 'r').read())
			name = file.split('/')[-1][:-5]
			pil_im = Image.open(os.path.join(self.path_images_train,name)).convert('RGB')
			im = np.asarray(pil_im)
			for d in range(len(data["Y"])):
				name_head = str(name_pic).zfill(10)+'.png'
				keypoints = data["X"][d]

				label = data["Y"][d]

				bbox = enlarge_bbox(data['bbox'][d])
				fullbody = crop_image(im, convert_bb(bbox))
				head = crop_head(im, keypoints, 'kitti')
				eyes = crop_eyes(im, keypoints, 'kitti')

				if idx_line > int(0.95*len(glob(os.path.join(self.path_annotation_train,'*.json')))):
					file_out.write(name+','+name_head+',val,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')
				else:
					file_out.write(name+','+name_head+',train,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')


				di = {"X":keypoints}
				json.dump(di, open(os.path.join(self.path_out,name_head+".json"), 'w'))

				Image.fromarray(fullbody).convert('RGB').save(os.path.join(self.path_out,'fullbodies/' + name_head))
				Image.fromarray(head).convert('RGB').save(os.path.join(self.path_out,'heads/' + name_head))
				Image.fromarray(eyes).convert('RGB').save(os.path.join(self.path_out,'eyes/' + name_head))

				name_pic += 1
"""
		for file in tqdm(glob(os.path.join(self.path_annotation_test,'*.json'))):
			data = json.loads(open(file, 'r').read())
			name = file.split('/')[-1][:-5]
			pil_im = Image.open(os.path.join(self.path_images_test,name)).convert('RGB')
			im = np.asarray(pil_im)
			for d in range(len(data["Y"])):
				name_head = str(name_pic).zfill(10)+'.png'
				keypoints = data["X"][d]

				label = data["Y"][d]

				bbox = enlarge_bbox(data['bbox'][d])
				fullbody = crop_image(im, convert_bb(bbox))
				head = crop_head(im, keypoints, 'kitti')
				eyes = crop_eyes(im, keypoints, 'kitti')

				file_out.write(name+','+name_head+',test,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')

				di = {"X":keypoints}
				json.dump(di, open(os.path.join(self.path_out,name_head+".json"), 'w'))

				Image.fromarray(fullbody).convert('RGB').save(os.path.join(self.path_out,'fullbodies/' + name_head))
				Image.fromarray(head).convert('RGB').save(os.path.join(self.path_out,'heads/' + name_head))
				Image.fromarray(eyes).convert('RGB').save(os.path.join(self.path_out,'eyes/' + name_head))

				name_pic += 1
		file_out.close()

class Kitti_label_loader():
	"""
		Class definition for Kitti gt annotation loader
	"""
	def __init__(self, path_anno_gt, path_anno, path_predicted):
		self.path_anno_gt = path_anno_gt
		self.path_anno = path_anno
		self.path_predicted = path_predicted

	def load_annotation_im_gt(self, im_name):
		txt_file = open(os.path.join(self.path_anno_gt, im_name[:-4]+'.txt'), 'r')
		bboxes = []
		for line in txt_file:
			line_s = line.split(" ")
			if line_s[0] == 'Pedestrian' or line_s[0] == "Cyclist":
				bb = [float(line_s[4]), float(line_s[5]), float(line_s[6]), float(line_s[7])]
				bboxes.append(bb)
		return bboxes

	def load_annotation_im(self, im, enlarge=1):
		new_data = []
		data = json.load(open(os.path.join(self.path_anno,im+'.predictions.json'), 'r'))
		data_ = [d["bbox"] for d in data]
		scores_ = [d["score"] for d in data]
		for d in data_:
			new_data.append(convert_bb(enlarge_bbox_kitti(d,enlarge)))
		return new_data, scores_

	def create_dicts(self, enlarge=1):
		di_gt = {}
		di_predicted = {}
		scores = {}
		for im in glob(os.path.join(self.path_predicted, '*.png')):
			if os.path.isfile(os.path.join(self.path_anno,im.split('/')[-1]+'.predictions.json')):
				name = im.split('/')[-1]
				if name not in di_gt:
					di_gt[name] = []
					di_predicted[name] = []
					scores[name] = []
				di_gt[name].extend(self.load_annotation_im_gt(name))
				pred_bb, pred_scores = self.load_annotation_im(name, enlarge)

				di_predicted[name].extend(pred_bb)
				scores[name].extend(pred_scores)
		return di_gt, di_predicted, scores


class AP_computer():
	"""
		Class definition to easily compute AP
	"""
	def __init__(self, di_gt, di_predicted_bboxes, di_scores, name='JAAD'):
		self.di_gt = di_gt
		self.di_predicted_bboxes = di_predicted_bboxes
		self.di_scores = di_scores

		self.path_out_txt = './input_{}'.format(name)
		self.path_out_txt_pred = os.path.join(self.path_out_txt, 'detection-results')
		self.path_out_txt_gt = os.path.join(self.path_out_txt, 'ground-truth')
		try:
			os.makedirs(self.path_out_txt)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

		try:
			os.makedirs(self.path_out_txt_pred)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

		try:
			os.makedirs(self.path_out_txt_gt)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise
	def create_txt(self):
		for name in self.di_predicted_bboxes.keys():
			name_processed = name.replace('/', '_')
			with open(os.path.join(self.path_out_txt_gt,name_processed+'.txt'), 'w') as file:
				bboxes = self.di_gt[name]
				for b in bboxes:
					file.write('person '+str(b[0])+' '+str(b[1])+' '+str(b[2])+' '+str(b[3])+'\n')
			file.close()
			with open(os.path.join(self.path_out_txt_pred, name_processed+'.txt'), 'w') as file:
				anno = self.di_predicted_bboxes[name]
				scores_ = self.di_scores[name]
				for i in range(len(anno)):
					file.write('person '+str(scores_[i])+' '+str(anno[i][0])+' '+str(anno[i][1])+' '+str(anno[i][2])+' '+str(anno[i][3])+'\n')
			file.close()
