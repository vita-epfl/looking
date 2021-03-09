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

def normalize(X, Y, divide=True):
    center_p = (int((X[11]+X[12])/2), int((Y[11]+Y[12])/2))
    X_new = np.array(X)
    Y_new = np.array(Y)-center_p[1]
    width = abs(np.max(X_new)-np.min(X_new))
    height = abs(np.max(Y_new)-np.min(Y_new))
    if divide:
        Y_new /= max(width, height)
        X_new /= max(width, height)
    return X_new, Y_new

class Example_Dataset(Dataset):
	"""Example dataset."""

	def __init__(self, split, path, dataset_type, transform=transforms.Compose([transforms.ToTensor()])):
		"""
		Dataset Loader class for a dataset. Args
		- split : either train or test
		- path: path to the image+instances folder
		- transform : transformations to apply on the images
		- dataset_type: str - jack / nu
		"""
		self.path = path
		self.split = split
		self.dataset_type = dataset_type
		self.transform = transform
		self.X, self.kps, self.Y = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		inp = Image.open(self.path+self.dataset_type+'/'+self.X[idx]+'.png') # You may need to change this line 
		sample = {'image': inp, 'keypoints':self.kps[idx], 'label': torch.Tensor([self.Y[idx]])}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], sample['keypoints'], sample['label']

	def preprocess(self):
		"""
		A function to parse the txt gt file and load the annotations, the labels and pre-loads the head images
		returns:
		- X : An array containing the image / joints name
		- Y : An array containing the labels
		- kps : A tensor containing all the joints for the corresponding set
		"""
		output = []
		X = []
		Y = []
		kps = []
		file = open('gt_{}.txt'.format(self.dataset_type), "r") # can change accordingly to where you save your txt file
		for line in file:
			line = line[:-1] # remove the \n
			line_s = line.split(",")
			if line_s[1] == self.split: # verify if it's the right split
				X.append(line_s[2]) # append the image / joints file name
				joints = np.array(json.load(open(self.path+self.dataset_type+'/'+line_s[2]))["X"])
				Xs = joints[:17] # take the Xs
				Ys = joints[17:34] # take the Ys
				X_new, Y_new = normalize(Xs, Ys, True) # normalize according to the paper
				tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist() # concatenate the instance to the array of joints
				kps.append(tensor) 
				Y.append(int(line_s[-1])) # append the label
		file.close()
		return X, torch.tensor(kps), Y