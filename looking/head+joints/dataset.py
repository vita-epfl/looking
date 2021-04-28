from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from glob import glob
from PIL import Image
from utils import *

np.random.seed(0)



class Kitti_Dataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, path, split, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = None
		self.path = path + 'Kitti/'
		self.transform = transform
		self.split = split
		assert self.split in ['train', 'test']
		self.X, self.kps, self.Y = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		inp = Image.open(self.path + self.X[idx])

		sample = {'image': inp, 'keypoints':self.kps[idx], 'label': torch.Tensor([self.Y[idx]])}

		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], sample['keypoints'], sample['label']

	def preprocess(self):
		output = []
		X = []
		Y = []
		kps = []
		file = open(self.path+'kitti_gt.txt', "r")
		for line in file:
			line = line[:-1]
			line_s = line.split(",")
			if line_s[2] == self.split:
				X.append(line_s[1])
				joints = np.array(json.load(open(self.path + line_s[1]+'.json'))["X"])
				Xs = joints[:17]
				Ys = joints[17:34]
				X_new, Y_new = normalize(Xs, Ys, True)
				tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
				kps.append(tensor)
				Y.append(int(line_s[-1]))
		file.close()
		return X, torch.tensor(kps), Y




class JAAD_Dataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, path, path_jaad, split, split_path, type_="video_new", transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = None
		self.path = path
		self.path_jaad = path_jaad
		self.split = split
		self.type = type_
		self.split_path = split_path
		assert self.type in ['original', 'video', 'ped', 'video_new']
		if self.type == 'video':
			self.txt = open(self.split_path + 'jaad_'+self.split+'_scenes_2k30.txt', "r")
		elif self.type == "original":
			self.txt  = open(self.split_path + 'jaad_'+self.split+'_original_2k30.txt', "r")
		self.transform = transform
		self.X, self.Y, self.kps = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.Y[idx]

		sample = {'image': Image.open(self.path+self.path_jaad+self.X[idx]), 'keypoints':self.kps[idx] ,'label':label}

		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], sample['keypoints'], sample['label']

	def preprocess(self):
		tab_X = []
		tab_Y = []
		kps = []
		for line in self.txt:
			line = line[:-1]
			line_s = line.split(",")
			tab_X.append(line_s[-2])
			joints = np.array(json.load(open(self.path+self.path_jaad+line_s[-2]+'.json'))["X"])
			X = joints[:17]
			Y = joints[17:34]
			X_new, Y_new = normalize(X, Y, True)
			tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
			kps.append(tensor)
			tab_Y.append(int(line_s[-1]))
		return tab_X, tab_Y, torch.tensor(kps)

	def evaluate(self, model, device, it=1):
		assert self.split in ["test", "val"]
		model.eval()
		model.to(device)
		tab_X, tab_Y, kps = self.X, self.Y, self.kps.cpu().detach().numpy()
		idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
		idx_Y0 = np.where(np.array(tab_Y) == 0)[0]

		positive_samples = np.array(tab_X)[idx_Y1]
		positive_samples_kps = np.array(kps)[idx_Y1]
		positive_samples_labels = np.array(tab_Y)[idx_Y1]
		N_pos = len(idx_Y1)
		aps = []
		accs = []

		for i in range(it):
			np.random.seed(i)
			np.random.shuffle(idx_Y0)

			neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
			neg_samples_kps = np.array(kps)[idx_Y0[:N_pos]]
			neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]

			total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
			total_samples_kps = np.concatenate((positive_samples_kps, neg_samples_kps)).tolist()
			total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()

			new_data = new_Dataset(self.path, self.path_jaad, total_samples, total_labels, total_samples_kps, self.transform)
			data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

			acc = 0
			out_lab = torch.Tensor([]).type(torch.float)
			test_lab = torch.Tensor([])
			for x_test, kps_test, y_test in data_loader:
				x_test, kps_test, y_test = x_test.to(device), kps_test.to(device), y_test.to(device)
				output = model(x_test, kps_test)
				out_pred = output
				pred_label = torch.round(out_pred)
				le = x_test.shape[0]
				test_lab = torch.cat((test_lab.detach().cpu(), y_test.view(-1).detach().cpu()), dim=0)
				out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)

			acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
			ap = average_precision(out_lab, test_lab)
			accs.append(acc.item())
			aps.append(ap)
		return np.mean(aps), np.mean(accs)


class new_Dataset(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path, path_jaad, data_x, data_y, kps, transform):
		"""
		Args:
			split : train, val and test
			type_ : type of dataset splitting (original splitting, video splitting, pedestrian splitting)
			transform : data tranformation to be applied
		"""
		self.data = None
		self.path = path
		self.path_jaad = path_jaad
		self.data_x = data_x
		self.data_y = data_y
		self.kps = kps
		self.transform = transform
	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.data_y[idx]
		label = torch.Tensor([label])
		sample = {'keypoints': self.kps[idx], 'image':Image.open(self.path+self.path_jaad+self.data_x[idx]),'label':label}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], torch.tensor(sample['keypoints']), sample['label']
