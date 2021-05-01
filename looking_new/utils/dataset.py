from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from glob import glob
from PIL import Image
from utils.utils_train import *

np.random.seed(0)


class JAAD_Dataset(Dataset):
    """Wrapper for JAAD Dataset (head, joints, head+joints)"""
    def __init__(self, path_data, type_='joints', split='train', pose="full", split_strategy="original", transform=None, path_txt='./splits_jaad'):
        self.path_data = path_data
        self.type = type_
        self.split = split
        self.split_strategy = split_strategy
        self.pose = pose
        self.transform = transform
        assert self.pose in ['full', 'head', 'body']
        assert self.split_strategy in ['instances', 'scenes']
        assert self.type in ['heads', 'joints']
        
        self.txt = open(os.path.join(path_txt, 'jaad_{}_{}.txt'.format(self.split, self.split_strategy)))
        self.Y, self.X = self.preprocess()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.Y[idx]

        if self.type == 'joints':
            sample = {'input':self.X[idx] ,'label':label}
        elif self.type == 'heads':
            sample = {'input': Image.open(os.path.join(self.path_data,self.X[idx])), 'label':label}
            if self.transform:
                sample['input'] = self.transform(sample['input'])
        return sample['input'], float(sample['label'])

    def get_joints(self):
        return self.kps, torch.Tensor(self.Y)

    def preprocess(self):
        if self.type == 'joints':
            tab_Y = []
            kps = []
            for line in self.txt:
                line = line[:-1]
                line_s = line.split(",")
                joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
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
            return tab_Y, torch.tensor(kps)
        elif self.type == 'heads':
            tab_X = []
            tab_Y = []
            for line in self.txt:
                line = line[:-1]
                line_s = line.split(",")
                tab_X.append(line_s[-2])
                tab_Y.append(int(line_s[-1]))
            return tab_Y, tab_X
    
    def evaluate(self, model, device, it=1):
        assert self.split in ["test", "val"]
        model = model.eval()
        model = model.to(device)
        if self.type == 'joints':
            
            tab_X, tab_Y = self.X.cpu().detach().numpy(), self.Y
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
                new_data = Eval_Dataset_joints(total_samples, total_labels)
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
        elif self.type == 'heads':
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
                #print(np.sum(total_labels))
                #print(len(total_labels))
                new_data = Eval_Dataset_heads(self.path_data, total_samples, total_labels, self.transform)
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
                #acc /= len(new_data)
                #rint(torch.round(out_lab).shape)
                #print(test_lab.shape)

                acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
                ap = average_precision(out_lab, test_lab)
                #print(ap)
                accs.append(acc.item())
                aps.append(ap)
            return np.mean(aps), np.mean(accs)

class Eval_Dataset_joints(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, data_x, data_y):
		"""
		Args:
			split : train, val and test
			type_ : type of dataset splitting (original splitting, video splitting, pedestrian splitting)
			transform : data tranformation to be applied
		"""
		self.data = None
		#self.path_jaad = path_jaad
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

class Eval_Dataset_heads(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path_jaad, data_x, data_y, transform=None):
		"""
		Args:
			split : train, val and test
			type_ : type of dataset splitting (original splitting, video splitting, pedestrian splitting)
			transform : data tranformation to be applied
		"""
		self.data = None
		self.path = path_jaad
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
		sample = {'image': Image.open(os.path.join(self.path, self.data_x[idx])), 'label':label}
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
