import os
import json
from glob import glob
import sys
from PIL import Image
import numpy as np

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

def main(argv):
	path_txt = './'
	train, test = parse_txt(path_txt, argv[1], argv[2])
	j = 0 
	if 'nu' in argv[3].lower():
		out = './nu/'
	elif 'jack'in argv[3].lower():
		out = './jack/'
	else:
		print('please use a valid dataset among [nu, jack]')


	if 'nu' in out:
		file_out = open('gt_nu.txt', 'w')
	elif 'jack' in out:
		file_out = open('gt_jack.txt', 'w')
	for raw in train:
		path = create_path(raw)
		for anno in glob(path+'/*.json'):
			name_im = raw+'/'+anno.split('/')[-1][:-5]
			im = Image.open(name_im)

			file = open(anno, 'r')
			json_di = json.load(file)
			for i in range(len(json_di["X"])):
				out_name = str(j).zfill(10)+'.json'

				bb = convert_bb(json_di["bbox"][i])
				head = crop(np.asarray(im), bb)
				Image.fromarray(head).convert('RGB').save(out+out_name+'.png')
				
				di = {"X":json_di["X"][i]}
				json.dump(di, open(out+out_name, "w"))
				line = ','.join([anno[:-5], 'train', out_name, str(json_di["bbox"][i][0]), str(json_di["bbox"][i][1]), str(json_di["bbox"][i][0]+json_di["bbox"][i][2]),str(json_di["bbox"][i][1]+json_di["bbox"][i][-1]), str(json_di["Y"][i])+'\n'])
				file_out.write(line)
				j += 1
			file.close()
	
	for raw in test:
		path = create_path(raw)
		#print(raw)
		for anno in glob(path+'/*.json'):
			name_im = raw+'/'+anno.split('/')[-1][:-5]
			im = Image.open(name_im)

			file = open(anno, 'r')
			json_di = json.load(file)
			for i in range(len(json_di["X"])):
				out_name = str(j).zfill(10)+'.json'

				bb = convert_bb(json_di["bbox"][i])
				head = crop(np.asarray(im), bb)
				Image.fromarray(head).convert('RGB').save(out+out_name+'.png')

				di = {"X":json_di["X"][i]}
				json.dump(di, open(out+out_name, "w"))
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
