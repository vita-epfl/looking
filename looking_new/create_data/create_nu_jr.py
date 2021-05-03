import os
import json
from glob import glob
import sys
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Creating Nuscenes and JR dataset')
parser.add_argument('--train_file', dest='trf', type=str, help='name of the train split file', default="trainvalJack.txt")
parser.add_argument('--test_file', dest='tsf', type=str, help='name of the test split file', default="testJack.txt")
parser.add_argument('--path_txt', dest='pt', type=str, help='path to the split folder', default="./splits_jack")
parser.add_argument('--name', dest='n', type=str, help='name of the dataset', default="jack")
parser.add_argument('--path_out', dest='po', type=str, help='path to the output files', default="/home/younesbelkada/Travail/data/JackRabbot")

args = parser.parse_args()


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

def crop(img, bbox):
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])

	x1, y1, x2, y2 = bbox
	h = y2-y1
	return img[int(y1):int(y1+(h/3)), x1:int(x2)]

def main(train_file, test_file, name_, out_txt, out, path_txt):
	train, test = parse_txt(path_txt, train_file, test_file, out)
	j = 0 

	assert name_.lower() in ['nu', 'jack']
	file_out = open(os.path.join(out_txt, 'ground_truth_{}.txt'.format(name_.lower())), 'w')
	for raw in train:
		path = create_path(raw)
		print(os.path.join(path,'*.json'))
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
				line = ','.join([anno[:-5], 'train', out_name, str(json_di["bbox"][i][0]), str(json_di["bbox"][i][1]), str(json_di["bbox"][i][0]+json_di["bbox"][i][2]),str(json_di["bbox"][i][1]+json_di["bbox"][i][-1]), str(json_di["Y"][i])+'\n'])
				file_out.write(line)
				j += 1
			file.close()
	
	for raw in test:
		path = create_path(raw)
		print(os.path.join(path,'*.json'))
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
			file.close()
	file_out.close()


path_txt = args.pt
path_out = args.po
test_file = args.tsf
train_file = args.trf
name = args.n

main(train_file, test_file, name, path_txt, path_out, path_txt)
