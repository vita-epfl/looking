import os
import json
from glob import glob
import sys
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Creating Nuscenes and JR dataset')
parser.add_argument('--train_file', dest='trf', type=str, help='name of the train split file', default="trainval.txt")
parser.add_argument('--test_file', dest='tsf', type=str, help='name of the test split file', default="test.txt")
parser.add_argument('--path_txt', dest='pt', type=str, help='path to the split folder', default="./splits_jack")
parser.add_argument('--name', dest='n', type=str, help='name of the dataset', default="jack")
parser.add_argument('--path_out', dest='po', type=str, help='path to the output files', default="/data/younes-data/JackRabbot")
parser.add_argument('--path_input', dest='pi', type=str, help='path to the input images', default="/data/younes-data/JackRabbot/reviewed")
parser.add_argument('--path_look_anno', dest='pla', type=str, help='path to the pifpaf anno files', default="/data/younes-data/LOOK_annotations/JackRabbot")

args = parser.parse_args()

#NU_TEST = ['n006', 'n008', 'n009', 'n014']
#NU_TRAIN = ['n003', 'n004', 'n005', 'n010', 'n013', 'n015', 'n016']

def parse_txt(path_txt, train_file, test_file, path_jack):
	"""
	Utility function to parse the txt files and retrive the paths
	"""
	train = []
	test = []
	with open(os.path.join(path_txt,train_file), 'r') as file:
		for line in file:
			train.append(os.path.join(path_jack, line[:-1]))
	with open(os.path.join(path_txt,test_file), 'r') as file:
		for line in file:
			test.append(os.path.join(path_jack, line[:-1]))
	return train, test

def create_path(raw_path):
	"""
	A simple utility function to calculate the path to annotations
	"""
	raw_s = raw_path.split('/')
	name = raw_s[-1]
	return raw_path+'/anno_'+name

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

def convert(data):
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

def crop(img, bbox):
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])

	x1, y1, x2, y2 = bbox
	h = y2-y1
	return img[int(y1):int(y1+(h/3)), x1:int(x2)]

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

def get_indice_max_iou(di_pifpaf, array_bboxes):
	bbox_pifpaf = convert_bb(di_pifpaf['bbox'])
	if len(array_bboxes) != 0:
		ious = np.array([bb_intersection_over_union(bbox_pifpaf, convert_bb(b)) for b in array_bboxes])
		if np.max(ious) >= 0.5:
			return np.where(ious == np.max(ious))[0][0]
	return -1

def main(train_file, test_file, name_, input_folder, out_txt, out, path_txt, path_anno, eyes_=False):
	train, test = parse_txt(path_txt, train_file, test_file, input_folder)
	train_anno, test_anno = parse_txt(path_txt, train_file, test_file, path_anno)
	file_out = open(os.path.join(out_txt, 'ground_truth_{}.txt'.format(name_.lower())), 'w')
	j = 0 
	assert name_.lower() in ['nu', 'jack', 'kitti']
	for i in range(len(train_anno)):
		raw_path = train_anno[i]
		raw_path_images = train[i]
		path_annotations = create_path(raw_path_images)
		for k, anno in enumerate(glob(os.path.join(raw_path,'*.json'))):
			name_im = os.path.join(raw_path_images, anno.split('/')[-1][:-17])
			

			file = open(anno, 'r')
			json_di = json.load(file)
			if os.path.isfile(os.path.join(path_annotations, anno.split('/')[-1][:-17]+'.json')):
				file_anno = open(os.path.join(path_annotations, anno.split('/')[-1][:-17]+'.json'), 'r')
				json_di_annotation = json.load(file_anno)
				bboxes_anno = json_di_annotation['bbox']
				for annotations_di in json_di:
					index_max_iou = get_indice_max_iou(annotations_di, bboxes_anno)
					looking_labels = json_di_annotation['Y']
					if index_max_iou != -1:
						out_name = str(j).zfill(10)+'.json'
						im = Image.open(name_im)
						bbox = convert_bb(enlarge_bbox(annotations_di['bbox']))
						converted_kps = convert(annotations_di['keypoints'])

						
						head = crop(np.asarray(im), bbox)
						
						
						Image.fromarray(head).convert('RGB').save(os.path.join(out,out_name+'.png'))
						if eyes_:
							eyes = crop_eyes(np.asarray(im), converted_kps)
							Image.fromarray(eyes).convert('RGB').save(os.path.join(out,out_name+'_eyes.png'))
						
							

						di = {"X":converted_kps}
						bbox_final_without_enlarge = convert_bb(annotations_di['bbox'])
						json.dump(di, open(os.path.join(out,out_name), "w"))
						#print(k)
						if k > int(0.9*len(glob(os.path.join(raw_path,'*.json')))):
							line = ','.join([name_im, 'val', out_name, str(bbox_final_without_enlarge[0]), str(bbox_final_without_enlarge[1]), str(bbox_final_without_enlarge[2]),str(bbox_final_without_enlarge[-1]), str(looking_labels[index_max_iou])+'\n'])
						else:
							line = ','.join([name_im, 'train', out_name, str(bbox_final_without_enlarge[0]), str(bbox_final_without_enlarge[1]), str(bbox_final_without_enlarge[2]),str(bbox_final_without_enlarge[-1]), str(looking_labels[index_max_iou])+'\n'])
						#print(line)
						file_out.write(line)
						j += 1
				file_anno.close()

			"""
			for i in range(len(json_di["X"])):
				out_name = str(j).zfill(10)+'.json'

				bb = convert_bb(enlarge_bbox(json_di["bbox"][i]))
				head = crop(np.asarray(im), bb)
				Image.fromarray(head).convert('RGB').save(os.path.join(out,out_name+'.png'))
				
				di = {"X":json_di["X"][i]}
				json.dump(di, open(os.path.join(out,out_name), "w"))
				if k > int(0.9*len(glob(os.path.join(path,'*.json')))):
				    line = ','.join([anno[:-5], 'val', out_name, str(json_di["bbox"][i][0]), str(json_di["bbox"][i][1]), str(json_di["bbox"][i][0]+json_di["bbox"][i][2]),str(json_di["bbox"][i][1]+json_di["bbox"][i][-1]), str(json_di["Y"][i])+'\n'])
				else:
				    line = ','.join([anno[:-5], 'train', out_name, str(json_di["bbox"][i][0]), str(json_di["bbox"][i][1]), str(json_di["bbox"][i][0]+json_di["bbox"][i][2]),str(json_di["bbox"][i][1]+json_di["bbox"][i][-1]), str(json_di["Y"][i])+'\n'])
				file_out.write(line)
				j += 1
			file.close()"""
		#exit(0)
	for i in range(len(test_anno)):
		raw_path = test_anno[i]
		raw_path_images = test[i]
		
		path_annotations = create_path(raw_path_images)
		#print(os.path.join(raw_path,'*.json'))
		#print(raw_path)
		#print(path_annotations)
		#exit(0)
		for k, anno in enumerate(glob(os.path.join(raw_path,'*.json'))):
			name_im = os.path.join(raw_path_images, anno.split('/')[-1][:-17])
			

			file = open(anno, 'r')
			json_di = json.load(file)
			#print(name_im)
			#print(os.path.join(path_annotations, anno.split('/')[-1][:-17]+'.json'))

			if os.path.isfile(os.path.join(path_annotations, anno.split('/')[-1][:-17]+'.json')):
				file_anno = open(os.path.join(path_annotations, anno.split('/')[-1][:-17]+'.json'), 'r')
				json_di_annotation = json.load(file_anno)
				bboxes_anno = json_di_annotation['bbox']
				for annotations_di in json_di:
					index_max_iou = get_indice_max_iou(annotations_di, bboxes_anno)
					looking_labels = json_di_annotation['Y']
					if index_max_iou != -1:
						out_name = str(j).zfill(10)+'.json'
						im = Image.open(name_im)
						bbox = convert_bb(enlarge_bbox(annotations_di['bbox']))
						head = crop(np.asarray(im), bbox)
						converted_kps = convert(annotations_di['keypoints'])
						Image.fromarray(head).convert('RGB').save(os.path.join(out,out_name+'.png'))
						if eyes_:
							eyes = crop_eyes(np.asarray(im), converted_kps)
							Image.fromarray(eyes).convert('RGB').save(os.path.join(out,out_name+'_eyes.png'))
						
							
						di = {"X":converted_kps}
						bbox_final_without_enlarge = convert_bb(annotations_di['bbox'])
						json.dump(di, open(os.path.join(out,out_name), "w"))
						#print(k)
						line = ','.join([name_im, 'test', out_name, str(bbox_final_without_enlarge[0]), str(bbox_final_without_enlarge[1]), str(bbox_final_without_enlarge[2]),str(bbox_final_without_enlarge[-1]), str(looking_labels[index_max_iou])+'\n'])
						#print(line)
						file_out.write(line)
						j += 1
				file_anno.close()
	"""for raw in test:
		path = create_path(raw)
		print(os.path.join(path,'*.json'))
		#print(tes
		for anno in glob(os.path.join(path,'*.json')):
			name_im = os.path.join(raw,anno.split('/')[-1][:-5])
			im = Image.open(name_im)

			file = open(anno, 'r')
			json_di = json.load(file)
			for i in range(len(json_di["X"])):
				out_name = str(j).zfill(10)+'.json'

				bb = convert_bb(enlarge_bbox(json_di["bbox"][i]))
				head = crop(np.asarray(im), bb)
				Image.fromarray(head).convert('RGB').save(os.path.join(out,out_name+'.png'))

				di = {"X":json_di["X"][i]}
				json.dump(di, open(os.path.join(out,out_name), "w"))
				line = ','.join([anno[:-5], 'test', out_name, str(json_di["bbox"][i][0]), str(json_di["bbox"][i][1]), str(json_di["bbox"][i][0]+json_di["bbox"][i][2]),str(json_di["bbox"][i][1]+json_di["bbox"][i][-1]), str(json_di["Y"][i])+'\n'])
				file_out.write(line)
				j += 1
			file.close()"""
	file_out.close()


path_txt = args.pt
path_out = args.po
test_file = args.tsf
train_file = args.trf
path_input = args.pi
name = args.n
path_anno_pifpaf = args.pla
main(train_file, test_file, name, path_input, path_txt, path_out, path_txt, path_anno_pifpaf, True)