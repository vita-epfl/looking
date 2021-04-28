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


class Kitti_Dataset_head(Dataset):
	"""Kitti Dataset loader for training & inference"""

	def __init__(self, path, split, transform=None):
		"""
		Args:
			split : train or test
			transform : image transformation to be applied on every instance
		"""
		self.data = None
		self.path = path + 'Kitti/'
		self.transform = transform
		self.split = split
		assert self.split in ['train', 'test']
		self.X, self.Y = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		"""
		Getter function to load the instances of the dataset
		"""
		if torch.is_tensor(idx):
			idx = idx.tolist()

		inp = Image.open(self.path + self.X[idx])
		sample = {'image': inp, 'label': torch.Tensor([self.Y[idx]])}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], sample['label']

	def preprocess(self):
		"""
		A helper function to load the paths of the head images
		"""
		output = []
		X = []
		Y = []
		file = open(self.path + 'kitti_gt.txt', "r")
		for line in file:
			line = line[:-1]
			line_s = line.split(",")
			if line_s[2] == self.split:
				X.append(line_s[1])
				Y.append(int(line_s[-1]))
		file.close()
		return X, Y


class JAAD_Dataset_head(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path, path_jaad, split, split_path, type_="original", transform=None):
		"""
		Args:
			split : train, val and test
			type_ : type of dataset splitting (original splitting, video splitting, pedestrian splitting)
			transform : data tranformation to be applied
		"""
		self.data = None
		self.path = path
		self.path_jaad = path_jaad
		self.split_path = split_path
		self.split = split
		self.type = type_
		assert self.type in ['original', 'video']

		if self.type == "video":
			self.txt = open(self.split_path + 'jaad_' + self.split + '_scenes_2k30.txt', "r")
		elif self.type == "original":
			self.txt = open(self.split_path + 'jaad_'+self.split + '_original_2k30.txt', "r")
		else:
			print("please select a valid type of split ")
			exit(0)

		self.transform = transform
		self.X, self.Y = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.Y[idx]
		label = torch.Tensor([label])
		sample = {'image': Image.open(self.path+self.path_jaad+self.X[idx]), 'label':label}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], sample['label']

	def evaluate(self, model, device, it):
		assert self.split in ["test", "val"]
		model.eval()
		print("Starting evalutation ..")
		tab_X, tab_Y = self.X, self.Y
		idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
		idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
		positive_samples = np.array(tab_X)[idx_Y1]
		positive_samples_labels = np.array(tab_Y)[idx_Y1]
		N_pos = len(idx_Y1)

		aps = []
		accs = []
		for i in range(it):
			np.random.seed(i)
			np.random.shuffle(idx_Y0)
			neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
			neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]

			total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
			total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()
			new_data = new_Dataset(self.path_jaad, total_samples, total_labels, self.transform)
			data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

			acc = 0
			out_lab = torch.Tensor([]).type(torch.float)
			test_lab = torch.Tensor([])
			for x_test, y_test in data_loader:
				x_test, y_test = x_test.to(device), y_test.to(device)
				output = model(x_test)
				out_pred = output
				pred_label = torch.round(out_pred)

				le = x_test.shape[0]
				acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
				test_lab = torch.cat((test_lab.detach().cpu(), y_test.view(-1).detach().cpu()), dim=0)
				out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)


			acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
			ap = average_precision(out_lab, test_lab)
			accs.append(acc.item())
			aps.append(ap)
		return np.mean(aps), np.mean(accs)

	def preprocess(self):
		"""
		A helper function to load the paths of the images from the txt file.
		"""
		tab_X = []
		tab_Y = []
		for line in self.txt:
			line = line[:-1]
			line_s = line.split(",")

			tab_X.append(line_s[-2])
			tab_Y.append(int(line_s[-1]))
		return tab_X, tab_Y


class new_Dataset(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path_jaad, data_x, data_y, transform=None):
		"""
		Args:
			split : train, val and test
			type_ : type of dataset splitting (original splitting, video splitting, pedestrian splitting)
			transform : data tranformation to be applied
		"""
		self.data = None
		self.path = path
		self.path_jaad = path_jaad
		self.transform = transform
		self.data_x = data_x
		self.data_y = data_y

	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.data_y[idx]
		label = torch.Tensor([label])
		sample = {'image': Image.open(self.path+self.path_jaad+self.data_x[idx]), 'label':label}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], sample['label']

	def preprocess(self):
		"""
		A helper function to load the paths of the images from the txt file.
		"""
		tab_X = []
		tab_Y = []
		for line in self.txt:
			line = line[:-1]
			line_s = line.split(",")
			tab_X.append(line_s[-2])
			tab_Y.append(int(line_s[-1]))
		return tab_X, tab_Y
