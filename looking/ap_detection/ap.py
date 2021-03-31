import os


def enlarge_bbox(bb, enlarge=3):
	delta_h = (bb[3]-bb[1]) / (5 * enlarge)
	delta_w = (bb[2]-bb[0]) / (10 * enlarge)
	bb[0] -= delta_w
	bb[1] -= delta_h
	bb[2] += delta_w
	bb[3] += delta_h
	return bb


def gt_to_data():
	file_gt = open("../results_new.csv", "r")
	data_gt = {}
	for line in file_gt:
		line_s = line[:-1].split(',')
		name = line_s[0]+'/'+line_s[2].zfill(5)+'.png'
		if name not in data_gt:
			data_gt[name] = []
			bb = [float(line_s[3]), float(line_s[4]), float(line_s[5]), float(line_s[6])]
			data_gt[name].append(bb)
		else:
			bb = [float(line_s[3]), float(line_s[4]), float(line_s[5]), float(line_s[6])]
			data_gt[name].append(bb)
		#break
	file_gt.close()
	return data_gt

def anno_to_data():
	data={}
	scores={}
	file_gt = open("jaad_test_scenes_2k30_ap.txt", "r")
	for line in file_gt:
		line_s = line[:-1].split(',')
		name = line_s[0]
		if name not in data:
			data[name] = []
			scores[name] = []
			bb = [float(line_s[2]), float(line_s[3]), float(line_s[4]), float(line_s[5])]
			bb = enlarge_bbox(bb)
			data[name].append(bb)
			scores[name].append(float(line_s[-2]))
		else:
			bb = [float(line_s[2]), float(line_s[3]), float(line_s[4]), float(line_s[5])]
			bb = enlarge_bbox(bb)
			data[name].append(bb)
			scores[name].append(float(line_s[-2]))
		#break
	file_gt.close()
	return data, scores

def write_to_txt(im, data_gt, data, scores):
	name = im.replace('/', '_')
	with open('./input/ground-truth/'+name+'.txt', 'w') as file:
		bboxes = data_gt[im]
		for b in bboxes:
			file.write('person '+str(b[0])+' '+str(b[1])+' '+str(b[2])+' '+str(b[3])+'\n')
	file.close()
	with open('./input/detection-results/'+name+'.txt', 'w') as file:
		anno = data[im]
		scores_ = scores[im]
		for i in range(len(anno)):
			file.write('person '+str(scores_[i])+' '+str(anno[i][0])+' '+str(anno[i][1])+' '+str(anno[i][2])+' '+str(anno[i][3])+'\n')
	file.close()

data_gt = gt_to_data()
data, scores = anno_to_data()

for key in data:
	write_to_txt(key, data_gt, data, scores)
