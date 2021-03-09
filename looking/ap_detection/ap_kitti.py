import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import matplotlib

import sklearn.metrics


path_anno_kitti = "/home/younesbelkada/Travail/Project/Kitti/others/data_object_label_2/training/label_2/"
path_anno_pifpaf = "/home/younesbelkada/Travail/Project/Kitti/annotated/Testset/"
path_anno = "/home/younesbelkada/Travail/Project/Kitti/annotated/v012_scale_2/"

def enlarge_bbox(bb, enlarge=1):
	delta_h = (bb[3]) / (7 * enlarge)
	delta_w = (bb[2]) / (3.5 * enlarge)
	bb[0] -= delta_w
	bb[1] -= delta_h
	bb[2] += delta_w
	bb[3] += delta_h
	return bb

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

def convert_bb(bb):
	return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]


def load_gt(im):
	#print(im)
	txt_file = open(path_anno_kitti+im[:-4]+'.txt', 'r')
	bboxes = []
	for line in txt_file:
		line_s = line.split(" ")
		if line_s[0] == 'Pedestrian' or line_s[0] == "Cyclist":
			bb = [float(line_s[4]), float(line_s[5]), float(line_s[6]), float(line_s[7])]
			bboxes.append(bb)
	return bboxes

def compute_ious(boxes_gt, boxes_pred):
	ious = []
	for b in boxes_pred:
		i = []
		for b_gt in boxes_gt:
			i.append(bb_intersection_over_union(b, b_gt))
		i = np.array(i)
		i[i< np.max(i)] = 0
		ious.append(i)
	return np.sum(np.array(ious), axis=0)
def load_anno(im):
	new_data = []
	data = json.load(open(path_anno+im+'.predictions.json', 'r'))
	data_ = [d["bbox"] for d in data]
	scores_ = [d["score"] for d in data]
	for d in data_:
		new_data.append(convert_bb(enlarge_bbox(d)))
	return new_data, scores_

def precision_recall(score, thresh=0.5):
	TP = np.sum(score >= thresh)
	FP = np.sum((score > 0) & (score < thresh))
	#FN = len(score)-TP
	#prec = TP/(TP+FP)
	#rec = TP/(TP+FN)
	return TP, FP

def get_all_pos():
	i = 0
	for im in os.listdir(path_anno_pifpaf):
		if im[-4:] == ".png":
			i += len(load_gt(im))
	return i

def write_to_txt(im, bboxes, anno, scores_):
	with open('./input/ground-truth/'+im[:-4]+'.txt', 'w') as file:
		for b in bboxes:
			file.write('person '+str(b[0])+' '+str(b[1])+' '+str(b[2])+' '+str(b[3])+'\n')
	file.close()
	with open('./input/detection-results/'+im[:-4]+'.txt', 'w') as file:
		for i in range(len(anno)):
			file.write('person '+str(scores_[i])+' '+str(anno[i][0])+' '+str(anno[i][1])+' '+str(anno[i][2])+' '+str(anno[i][3])+'\n')
	file.close()

predictions = []
y_true = []
for im in os.listdir(path_anno_pifpaf):
	if im[-4:] == ".png":
		bboxes = load_gt(im)
		anno, scores_ = load_anno(im)
		write_to_txt(im, bboxes, anno, scores_)
