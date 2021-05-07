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



class Kitti_Dataset_joints(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, path, split, pose="full"):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = None
		self.split = split
		self.pose = pose
		self.path = path + 'Kitti/'
		assert pose in ["head", 'full', 'body']
		assert self.split in ['train', 'test']
		self.X, self.Y = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		kps = self.X[idx, :]

		#normalized_kps = self.preprocess(kps)

		sample = {'image': kps, 'label': torch.Tensor([self.Y[idx]])}

		return sample['image'], sample['label']

	def get_joints(self):
		return self.X, self.Y

	def preprocess(self):
		output = []
		kps = []
		label = []
		file = open(self.path+'kitti_gt.txt', "r")
		for line in file:
			line = line[:-1]
			line_s = line.split(",")
			if line_s[2] == self.split:
				joints = np.array(json.load(open(self.path + line_s[1]+'.json'))["X"])
				X = joints[:17]
				Y = joints[17:34]
				X_new, Y_new = normalize(X, Y, True)
				if self.pose == "head":
					X_new, Y_new, C_new = extract_head(X_new, Y_new, joints[34:])
					tensor = np.concatenate((X_new, Y_new, C_new)).tolist()
				elif self.pose == 'body':
					X_new, Y_new, C_new = extract_body(X_new, Y_new, joints[34:])
					tensor = np.concatenate((X_new, Y_new, C_new)).tolist()
				else:
					tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
					tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
				kps.append(tensor)

				#X.append(line_s[1])
				label.append(int(line_s[-1]))
		file.close()
		return torch.Tensor(kps), torch.Tensor(label)


class JAAD_Dataset_joints(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, path, path_jaad, split, split_path, pose="full", type_="video", cluster=False):
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
		self.split_path = split_path
		self.split = split
		self.type = type_
		self.pose = pose
		self.cluster = cluster
		assert self.pose in ['full', 'head', 'body']
		assert self.type in ['original', 'video']
		if self.type == "video":
			self.txt = open(self.split_path + 'jaad_' + self.split + '_scenes_2k30.txt', "r")
		elif self.type == "original":
			self.txt = open(self.split_path + 'jaad_'+self.split + '_original_2k30.txt', "r")
		else:
			print("please select a valid type of split ")
			exit(0)
		self.Y, self.kps, self.filenames = self.preprocess()


	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.Y[idx]
		file_n = [self.files[idx]]

		sample = {'keypoints':self.kps[idx] ,'label':label, 'file_name': file_n}

		return sample['keypoints'], sample['label'], sample['file_name']

	def get_joints(self):
		return self.kps, torch.Tensor(self.Y)

	def equilibrate(self):
		np.random.seed(0)
		tab_X, tab_Y = self.kps.cpu().detach().numpy(), self.Y
		idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
		idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
		positive_samples = np.array(tab_X)[idx_Y1]
		positive_samples_labels = np.array(tab_Y)[idx_Y1]
		N_pos = len(idx_Y1)
		aps = []
		accs = []

		np.random.shuffle(idx_Y0)
		neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
		neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]
		total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
		total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()

		self.kps = torch.tensor(total_samples)
		self.Y = total_labels


	def preprocess(self):
		tab_Y = []
		kps = []
		files = []
		for line in self.txt:
			line = line[:-1]
			line_s = line.split(",")
			files.append(line_s[0])
			joints = np.array(json.load(open(self.path+self.path_jaad+line_s[-2]+'.json'))["X"])
			X = joints[:17]
			Y = joints[17:34]
			X_new, Y_new = normalize(X, Y, True)
			if self.pose == "head":
				X_new, Y_new, C_new = extract_head(X_new, Y_new, joints[34:])
				tensor = np.concatenate((X_new, Y_new, C_new)).tolist()
			elif self.pose == 'body':
				X_new, Y_new, C_new = extract_body(X_new, Y_new, joints[34:])
				tensor = np.concatenate((X_new, Y_new, C_new)).tolist()
			else:
				tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
			kps.append(tensor)
			tab_Y.append(int(line_s[-1]))
		return tab_Y, torch.tensor(kps), files

	def evaluate(self, model, device, it=1):
		assert self.split in ["test", "val"]
		model.eval()
		model.to(device)
		tab_X, tab_Y = self.kps.cpu().detach().numpy(), self.Y
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
			new_data = new_Dataset(total_samples, total_labels)
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
				acc += le*binary_acc(pred_label.type(torch.float), y_test.view(-1,1)).item()
				test_lab = torch.cat((test_lab.detach().cpu(), y_test.view(-1).detach().cpu()), dim=0)
				out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
			acc /= len(new_data)
			ap = average_precision(out_lab, test_lab)
			accs.append(acc)
			aps.append(ap)
		return np.mean(aps), np.mean(accs)

	def get_mislabeled_test(self, model, device):
		assert self.split in ["test"]
		model.eval()
		print("Starting evalutation ..")
		tab_X, tab_Y, filenames = self.kps.cpu().detach().numpy(), self.Y, self.filenames

		idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
		idx_Y0 = np.where(np.array(tab_Y) == 0)[0]

		positive_samples = np.array(tab_X)[idx_Y1]
		positive_samples_labels = np.array(tab_Y)[idx_Y1]
		pos_files = np.array(filenames)[idx_Y1]
		N_pos = len(idx_Y1)

		aps = []
		accs = []
		np.random.seed(0)
		np.random.shuffle(idx_Y0)
		neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
		neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]
		neg_files = np.array(filenames)[idx_Y0[:N_pos]]

		total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
		total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()
		total_filenames = np.concatenate((pos_files, neg_files)).tolist()

		new_data = new_Dataset_qualitative(self.path, self.path_jaad, total_samples, total_labels, total_filenames)
		data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

		acc = 0
		false_neg, false_pos = [], []
		out_lab = torch.Tensor([]).type(torch.float)
		test_lab = torch.Tensor([])
		for x_test, y_test, f_name in data_loader:
			x_test, y_test = x_test.to(device), y_test.to(device)
			output = model(x_test)
			out_pred = output
			pred_label = torch.round(out_pred)

			if y_test == 1 and pred_label == 0:
				# False negative
				false_neg.append([f_name, pred_label])
			elif y_test == 0 and pred_label == 1:
				# False postitve
				false_pos.append([f_name, pred_label])

			le = x_test.shape[0]
			acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
			test_lab = torch.cat((test_lab.detach().cpu(), y_test.view(-1).detach().cpu()), dim=0)
			out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)


		acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
		ap = average_precision(out_lab, test_lab)
		accs.append(acc.item())
		aps.append(ap)
		return np.mean(aps), np.mean(accs), false_pos, false_neg


class new_Dataset(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, data_x, data_y):
		"""
		Args:
			split : train, val and test
			type_ : type of dataset splitting (original splitting, video splitting, pedestrian splitting)
			transform : data tranformation to be applied
		"""
		self.data = None
		self.data_x = data_x
		self.data_y = data_y

	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.data_y[idx]
		label = torch.Tensor([label])
		sample = {'image': self.data_x[idx], 'label':label}
		return torch.Tensor(sample['image']), sample['label']


class new_Dataset_qualitative(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path, path_jaad, data_x, data_y, files):
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
		self.files = files

	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		file_n = [self.files[idx]]
		label = self.data_y[idx]
		label = torch.Tensor([label])
		sample = {'image': Image.open(self.path+self.path_jaad+self.data_x[idx]), 'label': label, 'file_name': file_n}

		return sample['image'], sample['label'], sample['file_name']
