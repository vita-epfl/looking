
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

def crop_kitti(img, bbox):
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])
	x1, y1, w, h = bbox
	h /= 3
	return img[int(y1):int(y1+h), x1:int(x1+w)]


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

def crop_jaad(img, bbox):
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])

	x1, y1, x2, y2 = bbox
	h = y2-y1
	return img[int(y1):int(y1+(h/3)), x1:int(x2)]

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
	def __init__(self, path_jaad):
		self.imdb = JAAD(data_path=path_jaad)
		self.imdb.generate_database()
		self.data = pickle.load(open(os.path.join(path_jaad,'data_cache/jaad_database.pkl'), 'rb'))
	
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

class JAAD_creator():
	"""
		Class definition for creating our custom JAAD dataset for heads / joints / heads+joints training 
	"""
	def __init__(self, txt_out, dir_out, path_joints, path_jaad):
		self.txt_out = txt_out
		self.dir_out = dir_out
		self.path_joints = path_joints
		self.path_jaad = path_jaad
	
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
					head = crop_jaad(img, bb_final)
					
					Image.fromarray(head).convert('RGB').save(os.path.join(self.dir_out,name_head))
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
		for idx_line, file in enumerate(tqdm(glob(os.path.join(self.path_annotation_train,'*.json')))):
			data = json.loads(open(file, 'r').read())
			name = file.split('/')[-1][:-5]
			pil_im = Image.open(os.path.join(self.path_images_train,name)).convert('RGB')
			im = np.asarray(pil_im)
			for d in range(len(data["Y"])):
				name_head = str(name_pic).zfill(10)+'.png'
				keypoints = data["X"][d]
				
				label = data["Y"][d]
				
				bbox = enlarge_bbox(data['bbox'][d])
				head = crop_kitti(im, bbox)

				if idx_line > int(0.95*len(glob(os.path.join(self.path_annotation_train,'*.json')))):
					file_out.write(name+','+name_head+',val,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')
				else:
					file_out.write(name+','+name_head+',train,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')

				di = {"X":keypoints}
				json.dump(di, open(os.path.join(self.path_out,name_head+".json"), 'w'))

				Image.fromarray(head).convert('RGB').save(os.path.join(self.path_out,name_head))
				name_pic += 1
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
				head = crop_kitti(im, bbox)

				file_out.write(name+','+name_head+',test,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')

				di = {"X":keypoints}
				json.dump(di, open(os.path.join(self.path_out,name_head+".json"), 'w'))

				Image.fromarray(head).convert('RGB').save(os.path.join(self.path_out,name_head))
				name_pic += 1
		file_out.close()
	
