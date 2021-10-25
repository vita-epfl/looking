from pie_data import PIE
import os
import numpy as np
import pickle
import json
import errno
from glob import glob
from PIL import Image, ImageFile
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Script to build the PIE dataset')
parser.add_argument('--path_pie', dest='ppie', type=str, help='path to the PIE dataset', default="./PIE")
parser.add_argument('--path_out_txt', dest='pot', type=str, help='path to the output txt files', default="./splits_pie")
parser.add_argument('--path_output_files', dest='pof', type=str, help='path to the output images', default="/data/younes-data/PIE")
parser.add_argument('--path_keypoints', dest='pkps', type=str, help='path to the pifpaf files', default="/data/younes-data")

args = parser.parse_args()

np.random.seed(0)

ImageFile.LOAD_TRUNCATED_IMAGES = True

## Helper functions

def crop_eyes(img, keypoints_array_, eps=2):
	"""
		Crop the eye region given the crop of the head. 
		Args:
			- image : PIL Image object of the head region
			- keypoints : Tensor object of the keypoints
		Returns:
			the cropped eye region
	"""
	keypoints_array = keypoints_array_.copy()
	for i in range(len(keypoints_array)):
		if keypoints_array[i] < 0:
			keypoints_array[i] = 0
	img_width, img_height = img.shape[1], img.shape[0]

	left_most_point = min(keypoints_array[3], keypoints_array[4], keypoints_array[0], keypoints_array[1], keypoints_array[2])
	right_most_point = max(keypoints_array[3], keypoints_array[4], keypoints_array[0], keypoints_array[1], keypoints_array[2])


	top_most_point = min(keypoints_array[21], keypoints_array[20], keypoints_array[17], keypoints_array[18], keypoints_array[19])
	bottom_most_point = max(keypoints_array[21], keypoints_array[20], keypoints_array[17], keypoints_array[18], keypoints_array[19])
	
	
	x1, y1 = int(left_most_point-eps), int(top_most_point-eps)
	x2, y2 = int(right_most_point+eps), int(bottom_most_point+eps)

	#print(keypoints[18]-keypoints[19])

	if x1 < 0:
		x1 = 0
	if x2 <= 0:
		x2 = 5
	if y1 < 0:
		y1 = 0
	if y2 <= 0:
		y2 = 5

	if x1 >= img_width:
		x1 = img_width-5
	if x2 >= img_width:
		x2 = img_width
	if y1 >= img_height:
		y1 = img_height-5
	if y2 >= img_height:
		y2 = img_height

	return img[y1:y2, x1:x2]

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

def create_directory(dir_name):
	try:
		os.makedirs(dir_name)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

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
class PIE_loader():
    def __init__(self, path_pie):
        self.path_pie = path_pie
        self.imdb = PIE(data_path=path_pie)
        self.imdb.generate_database()
        self.data = pickle.load(open(os.path.join(self.path_pie,'data_cache/pie_database.pkl'), 'rb'))
    def generate(self):
        """
            Generate the annotations in a dictionary
        """
        data = {"path":[], "bbox":[], "names":[], "Y":[], "video":[]}
        for sets in self.data.keys():
            for videos in self.data[sets].keys():
                ped_anno = self.data[sets][videos]["ped_annotations"]
                for d in ped_anno:
                    if 'look' in ped_anno[d]['behavior'].keys():
                        for i in range(len(ped_anno[d]['frames'])):

                            path = os.path.join(sets, videos,str(ped_anno[d]['frames'][i]).zfill(5)+'.png')
                            bbox = [ped_anno[d]['bbox'][i][0], ped_anno[d]['bbox'][i][1], ped_anno[d]['bbox'][i][2], ped_anno[d]['bbox'][i][3]]
                            
                            data['path'].append(path)
                            data["bbox"].append(bbox)
                            data['Y'].append(ped_anno[d]['behavior']['look'][i])
                            data['names'].append(d)
                            data["video"].append(sets+'_'+videos)
        return data

class PIE_creator():
    """
        Class definition for creating our custom PIE dataset for heads / joints / heads+joints training 
    """
    def __init__(self, txt_out, dir_out, path_joints, path_pie):
        self.txt_out = txt_out
        self.dir_out = dir_out
        self.path_joints = path_joints
        self.path_jaad = path_pie
    
    def create(self, dict_annotation):
        """
            Create the files given the dictionary with the annotations 
        """

        file_out = open(os.path.join(self.txt_out, "ground_truth.txt"), "w+")
        name_pic = 0
        for i in range(len(dict_annotation["Y"])):
            path = dict_annotation["path"][i]
            print(os.path.join(self.path_joints,path+'.predictions.json'))
            #exit(0)
            if os.path.isfile(os.path.join(self.path_joints,path+'.predictions.json')):
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
                        name_eyes = str(name_pic).zfill(10)+'_eyes.png'
                        img = Image.open(os.path.join(self.path_jaad,path)).convert('RGB')
                        img = np.asarray(img)
                        head = crop_jaad(img, bb_final)
                        eyes = crop_eyes(img, kps)
                        
                        Image.fromarray(head).convert('RGB').save(os.path.join(self.dir_out,name_head))
                        Image.fromarray(eyes).convert('RGB').save(os.path.join(self.dir_out,name_eyes))
                        kps_out = {"X":kps}
                        
                        json.dump(kps_out, open(os.path.join(self.dir_out,name_head+'.json'), "w"))
                        line = path+','+dict_annotation["names"][i]+','+str(bb_final[0])+','+str(bb_final[1])+','+str(bb_final[2])+','+str(bb_final[3])+','+str(sc)+','+str(iou_max)+','+name_head+','+str(dict_annotation["Y"][i])+'\n'
                        name_pic += 1
                        file_out.write(line)
        file_out.close()

path_pie = args.ppie
path_output_txt = args.pot
path_output_files = args.pof
path_keypoints = args.pkps

create_directory(path_output_txt)
create_directory(path_output_files)

pie_loader = PIE_loader(path_pie)
data = pie_loader.generate()

creator = PIE_creator(path_output_txt, path_output_files, path_keypoints, path_pie)
creator.create(data)
