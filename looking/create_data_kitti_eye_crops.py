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


def eye_crops(img, kps):
    default_l = 10
    idx = [1,2]
    candidate1, candidate2 = kps[3*idx[0]:3*idx[0]+2], kps[3*idx[1]:3*idx[1]+2]
    #candidate1, candidate2 = [kps[idx[0]], kps[17+idx[0]], kps[34+idx[0]]], [kps[idx[1]], kps[17+idx[1]], kps[34+idx[1]]]
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
        bbox = [kps1[0]-0.9*max(l, default_l), kps1[1]-0.9*max(l, default_l), kps2[0]+0.9*max(l, default_l), kps2[1]+0.5*max(l, default_l)]
        #bbox = [kps1[0]-1.2*max(l, 5), kps1[1]-0.8*max(l, 5), kps2[0]+1.2*max(l, 5), kps2[1]+0.8*max(l, 5)]
        for i in range(len(bbox)):
            if bbox[i] < 0:
                bbox[i] = 0
            else:
                bbox[i] = int(bbox[i])

        x1, y1, x2, y2 = bbox

        # No eyes in picture
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
        # Eyes not in pic: return black image
        not_in_pic = img.copy()
        not_in_pic.fill(0)
        return not_in_pic[0:14, 0:25]

train = True
test = True

if train:
	path_train_images = '/scratch/izar/caristan/data/Kitti/Trainset/'
	path_train = '/scratch/izar/caristan/data/Kitti/Trainset/anno_Trainset/'
	final_di = {'eyes':[], 'keypoints':[], 'Y':[], 'names':[]}
	path_out = "/scratch/izar/caristan/data/Kitti_eye_crops/"

	name_pic = 0
	file_out = open("kitti_gt_eyes_crops.txt", "w")
	for file in tqdm(glob(path_train+'*.json')):
		data = json.loads(open(file, 'r').read())
		name = file.split('/')[-1][:-5]
		pil_im = PIL.Image.open(path_train_images+name).convert('RGB')
		im = np.asarray(pil_im)
		for d in range(len(data["Y"])):
			name_eyes = str(name_pic).zfill(10)+'.png'
			keypoints = data["X"][d]

			label = data["Y"][d]

			bbox = enlarge_bbox(data['bbox'][d])
			eyes = eye_crops(im, keypoints)
			file_out.write(name+','+name_eyes+',train,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')

			di = {"X":keypoints}
			json.dump(di, open(path_out+name_eyes+".json", 'w'))

			Image.fromarray(eyes).convert('RGB').save(path_out+name_eyes)
			name_pic += 1

	#json.dump(final_di, open("kitti_eyes_joints_train.json", 'w'))

if test:
	path_train_images = '/scratch/izar/caristan/data/Kitti/Testset/'
	path_train = '/scratch/izar/caristan/data/Kitti/Testset/anno_Testset/'

	final_di = {'eyes':[], 'keypoints':[], 'Y':[], 'names':[]}


	for file in tqdm(glob(path_train+'*.json')):
		data = json.loads(open(file, 'r').read())
		name = file.split('/')[-1][:-5]
		pil_im = PIL.Image.open(path_train_images+name).convert('RGB')
		im = np.asarray(pil_im)
		for d in range(len(data["Y"])):
			name_eyes = str(name_pic).zfill(10)+'.png'
			keypoints = data["X"][d]

			label = data["Y"][d]

			bbox = enlarge_bbox(data['bbox'][d])
			eyes = eye_crops(im, keypoints)

			file_out.write(name+','+name_eyes+',test,'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])+','+str(label)+'\n')
			di = {"X":keypoints}
			json.dump(di, open(path_out+name_eyes+".json", 'w'))

			Image.fromarray(eyes).convert('RGB').save(path_out+name_eyes)
			name_pic += 1



file_out.close()
