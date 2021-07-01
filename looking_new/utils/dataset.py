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
	"""Wrapper for JAAD Dataset (head, joints, head+joints, ...)"""
	def __init__(self, path_data, type_='joints', split='train', pose="full", split_strategy="original", transform=None, path_txt='./splits_jaad', device=torch.device('cpu')):
		self.path_data = path_data
		self.type = type_
		self.split = split
		self.split_strategy = split_strategy
		self.pose = pose
		self.transform = transform
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.name = 'jaad'
		assert self.pose in ['full', 'head', 'body']
		assert self.split_strategy in ['instances', 'scenes']
		assert self.type in ['joints', 'heads', 'eyes', 'fullbodies', 'heads+joints', 'eyes+joints', 'fullbodies+joints']

		self.txt = open(os.path.join(path_txt, 'jaad_{}_{}.txt'.format(self.split, self.split_strategy)), 'r')

		if '+' in self.type:
			self.X, self.kps, self.Y = self.preprocess()
		else:
			self.X, self.Y = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.Y[idx]

		if self.type == 'joints':
			sample = {'input':self.X[idx].to(self.device) ,'label':label}
		elif self.type == 'eyes':
			sample = {'input': Image.open(os.path.join(self.path_data, 'eyes/'+self.X[idx])), 'label':label}
			if self.transform:
				sample['input'] = torch.flatten(self.transform(sample['input']))
			else:
				sample['input'] = torch.flatten(sample['input'])
		# Image crop
		elif '+' not in self.type:
			sample = {'input': Image.open(os.path.join(self.path_data, self.type+'/'+self.X[idx])), 'label':label}
			if self.transform:
				sample['input'] = self.transform(sample['input']).to(self.device)
		# Multimodel: crop+joints_type
		else:
			sample_ = {'image': Image.open(os.path.join(self.path_data,self.type.split('+')[0]+'/'+self.X[idx])), 'keypoints':self.kps[idx] ,'label':label}
			if self.transform:
				sample_['image'] = self.transform(sample_['image'])
			sample = {'input':(sample_['image'].to(self.device), sample_['keypoints'].to(self.device)), 'label':sample_['label']}
		return sample['input'], torch.Tensor([float(sample['label'])])

	def get_joints(self):
		return self.X, torch.Tensor(self.Y).unsqueeze(1)

	def preprocess(self):
		self.heights = []
		if self.type == 'joints':
			tab_Y = []
			kps = []
			for line in self.txt:
				line = line[:-1]
				line_s = line.split(",")
				joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
				X = joints[:17]
				Y = joints[17:34]
				X_new, Y_new, height = normalize(X, Y, divide=True, height_=True)
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
				self.heights.append(height)
			return torch.tensor(kps), tab_Y
		elif '+' not in self.type:
			tab_X = []
			tab_Y = []
			for line in self.txt:
				line = line[:-1]
				line_s = line.split(",")
				tab_X.append(line_s[-2])
				tab_Y.append(int(line_s[-1]))
				joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
				X = joints[:17]
				Y = joints[17:34]
				X_new, Y_new, height = normalize(X, Y, divide=True, height_=True)
				self.heights.append(height)
			return tab_X, tab_Y
		else:
			tab_X = []
			tab_Y = []
			kps = []
			for line in self.txt:
				line = line[:-1]
				line_s = line.split(",")
				tab_X.append(line_s[-2])
				joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
				X = joints[:17]
				Y = joints[17:34]
				X_new, Y_new = normalize(X, Y, True)
				tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
				kps.append(tensor)
				tab_Y.append(int(line_s[-1]))
			return tab_X, torch.tensor(kps), tab_Y

	def eval_ablation(self, heights, preds, ground_truths):
		"""
			Evaluate using the ablation study
		"""
		preds = preds.detach().cpu().numpy()
		ground_truths = ground_truths.detach().cpu().numpy()

		#perc_1 = np.percentile(heights, 5)
		#perc_2 = np.percentile(heights, 50)
		#perc_3 = np.percentile(heights, 95)
		perc_1 = 65
		perc_2 = 165
		perc_3 = 365


		preds_1 = preds[heights <= perc_1]
		ground_truths_1 = ground_truths[heights <= perc_1]

		preds_2 = preds[(heights > perc_1) & (heights <= perc_2)]
		ground_truths_2 = ground_truths[(heights > perc_1) & (heights <= perc_2)]

		preds_3 = preds[(heights > perc_2) & (heights <= perc_3)]
		ground_truths_3 = ground_truths[(heights > perc_2) & (heights <= perc_3)]

		preds_4 = preds[heights > perc_3]
		ground_truths_4 = ground_truths[heights > perc_3]

		ap_1, ap_2, ap_3, ap_4 = average_precision_score(ground_truths_1, preds_1), average_precision_score(ground_truths_2, preds_2), average_precision_score(ground_truths_3, preds_3), average_precision_score(ground_truths_4, preds_4)
		ac_1, ac_2, ac_3, ac_4 = get_acc_per_distance(ground_truths_1, preds_1), get_acc_per_distance(ground_truths_2, preds_2), get_acc_per_distance(ground_truths_3, preds_3), get_acc_per_distance(ground_truths_4, preds_4)

		#print('Far :{} | Middle : {} | Close : {}'.format(ap_1, ap_2, ap_3))

		distances = [perc_1, perc_2, perc_3]
		#print(distances)
		#exit(0)

		return (ap_1, ap_2, ap_3, ap_4), (ac_1, ac_2, ac_3, ac_4), distances


	def evaluate(self, model, device, it=1, heights_=False):
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
			#print(N_pos)
			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			acs1 = []
			acs2 = []
			acs3 = []
			acs4 = []
			distances = []
			for i in range(it):
				np.random.seed(i)
				np.random.shuffle(idx_Y0)
				neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
				neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]


				total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
				total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()

				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

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
					acc += le*binary_acc(pred_label.type(torch.float), y_test.type(torch.float).view(-1)).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
				if heights_:
					out_ap, out_ac, distance = self.eval_ablation(heights, out_lab, test_lab)
					ap_1, ap_2, ap_3, ap_4 = out_ap
					ac_1, ac_2, ac_3, ac_4 = out_ac

					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)

					acs1.append(ac_1)
					acs2.append(ac_2)
					acs3.append(ac_3)
					acs4.append(ac_4)
					distances.append(distance)
					#break



				acc /= len(new_data)
				ap = average_precision(out_lab, test_lab)


				accs.append(acc)
				aps.append(ap)
			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			return np.mean(aps), np.mean(accs)
		elif '+' not in self.type:
			tab_X, tab_Y = self.X, self.Y
			idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
			idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
			positive_samples = np.array(tab_X)[idx_Y1]
			positive_samples_labels = np.array(tab_Y)[idx_Y1]
			N_pos = len(idx_Y1)

			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			acs1 = []
			acs2 = []
			acs3 = []
			acs4 = []
			distances = []
			for i in range(it):
				np.random.seed(i)
				np.random.shuffle(idx_Y0)
				neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
				neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]

				total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
				total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()
				new_data = Eval_Dataset_crop(self.path_data, total_samples, total_labels, self.type, self.transform)
				data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)


				acc = 0
				out_lab = torch.Tensor([]).type(torch.float)
				test_lab = torch.Tensor([])
				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

				for x_test, y_test in data_loader:
					x_test, y_test = x_test.to(device), y_test.to(device)
					output = model(x_test)
					out_pred = output
					pred_label = torch.round(out_pred)

					le = x_test.shape[0]
					acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)


				acc /= len(new_data)
				#rint(torch.round(out_lab).shape)
				#print(test_lab.shape)

				#acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
				#print(acc)
				ap = average_precision(out_lab, test_lab)
				#print(ap)
				accs.append(acc)
				aps.append(ap)
				if heights_:
					#ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
					out_ap, out_ac, distance = self.eval_ablation(heights, out_lab, test_lab)
					ap_1, ap_2, ap_3, ap_4 = out_ap
					ac_1, ac_2, ac_3, ac_4 = out_ac

					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)

					acs1.append(ac_1)
					acs2.append(ac_2)
					acs3.append(ac_3)
					acs4.append(ac_4)
					distances.append(distance)
				#break
			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			return np.mean(aps), np.mean(accs)
		else:
			model = model.eval()
			model = model.to(device)
			#print("Starting evalutation ..")
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

				new_data = Eval_Dataset_multimodel(self.path_data, total_samples, total_labels, total_samples_kps, self.type, self.transform)
				data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

				acc = 0
				out_lab = torch.Tensor([]).type(torch.float)
				test_lab = torch.Tensor([])
				for x_test, kps_test, y_test in data_loader:
					x_test, kps_test, y_test = x_test.to(device), kps_test.to(device), y_test.to(device)
					output = model((x_test, kps_test))
					out_pred = output
					pred_label = torch.round(out_pred)
					le = x_test.shape[0]
					#print(test_lab)
					#print(y_test)
					acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)

				acc /= len(new_data)
				#acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)

				ap = average_precision(out_lab, test_lab)
				#print(ap)
				accs.append(acc)
				aps.append(ap)
			return np.mean(aps), np.mean(accs)

class Eval_Dataset_joints(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, data_x, data_y, heights=None):
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
		self.heights = heights
		self.type = type

	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.data_y[idx]
		sample = {'image': self.data_x[idx], 'label':label}
		return torch.Tensor(sample['image']), sample['label']

class Eval_Dataset_crop(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path_jaad, data_x, data_y, type, transform=None, heights=None):
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
		self.heights = heights
		self.type = type


	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.data_y[idx]
		sample = {'image': Image.open(os.path.join(self.path, self.type + '/' + self.data_x[idx])), 'label':label}
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

class Eval_Dataset_multimodel(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path_data, data_x, data_y, kps, type, transform, heights=None):
		"""
		Args:
			split : train, val and test
			type_ : type of dataset splitting (original splitting, video splitting, pedestrian splitting)
			transform : data tranformation to be applied
		"""
		self.data = None
		self.path_data = path_data
		self.data_x = data_x
		self.data_y = data_y
		self.kps = kps
		self.transform = transform
		self.heights = heights
		self.type = type

	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.data_y[idx]
		sample = {'keypoints': self.kps[idx], 'image':Image.open(os.path.join(self.path_data, self.type + '/' + self.data_x[idx])),'label':label}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
		return sample['image'], torch.tensor(sample['keypoints']), sample['label']

class Kitti_dataset(Dataset):
	"""
		Class definition for Kitti dataset (Head / joints / Head+joints / ...)
	"""
	def __init__(self, split, type_, path_gt_txt, path_data, pose="full", transform=None, device=torch.device('cpu')):
		"""
		"""
		self.name = 'kitti'
		self.data = None
		self.path_data = path_data
		self.path_gt_txt = path_gt_txt
		self.transform = transform
		self.split = split
		self.pose = pose
		self.type = type_
		self.device = device
		assert self.type in ['joints', 'heads', 'eyes', 'fullbodies', 'heads+joints', 'eyes+joints', 'fullbodies+joints']
		assert self.pose in ["head", 'full', 'body']
		assert self.split in ['train', 'test', 'val']
		self.file  = open(os.path.join(self.path_gt_txt,'ground_truth_kitti.txt'), "r")

		if '+' in self.type:
			self.X, self.kps, self.Y = self.preprocess()
		else:
			self.X, self.Y = self.preprocess()

	def __len__(self):
		return len(self.Y)

	def get_joints(self):
		return self.X, self.Y

	def preprocess(self):
		self.heights = []
		if self.type == 'joints':
			output = []
			kps = []
			label = []
			for line in self.file:
				line = line[:-1]
				line_s = line.split(",")
				if line_s[2] == self.split:
					joints = np.array(json.load(open(os.path.join(self.path_data,line_s[1]+'.json')))["X"])
					X = joints[:17]
					Y = joints[17:34]
					X_new, Y_new = normalize(X, Y, divide=True, height_=False)
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
					label.append(int(line_s[-1]))
					self.heights.append(int(float(line_s[-2])))
			self.file.close()
			return torch.Tensor(kps), torch.Tensor(label)
		elif '+' not in self.type:
			output = []
			X = []
			Y = []
			for line in self.file:
				line = line[:-1]
				line_s = line.split(",")
				if line_s[2] == self.split:
					X.append(line_s[1])
					Y.append(int(line_s[-1]))
					self.heights.append(int(float(line_s[-2])))
			self.file.close()
			return X, Y
		else:
			output = []
			X = []
			Y = []
			kps = []
			for line in self.file:
				line = line[:-1]
				line_s = line.split(",")
				if line_s[2] == self.split:
					X.append(line_s[1])
					joints = joints = np.array(json.load(open(os.path.join(self.path_data,line_s[1]+'.json')))["X"])
					Xs = joints[:17]
					Ys = joints[17:34]
					X_new, Y_new = normalize(Xs, Ys, True)
					tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
					kps.append(tensor)
					Y.append(int(line_s[-1]))
					self.heights.append(int(float(line_s[-2])))
			self.file.close()
			return X, torch.tensor(kps), Y

	def __getitem__(self, idx):
		"""
		Getter function to load the instances of the dataset
		"""
		if torch.is_tensor(idx):
			idx = idx.tolist()

		label = self.Y[idx]
		if self.type == 'joints':
			kps = self.X[idx, :]
			sample = {'image': kps, 'label': label}
			return sample['image'].to(self.device), torch.Tensor([float(sample['label'])])
		elif '+' not in self.type:
			inp = Image.open(os.path.join(self.path_data, self.type + '/' + self.X[idx]))
			sample = {'image': inp, 'label': label}
			if self.transform:
				sample['image'] = self.transform(sample['image'])
			return sample['image'].to(self.device), torch.Tensor([float(sample['label'])])
		else:
			inp = Image.open(os.path.join(self.path_data, self.type.split('+')[0] + '/' + self.X[idx]))
			sample = {'image': inp, 'keypoints':self.kps[idx], 'label': label}
			if self.transform:
				sample['image'] = self.transform(sample['image'])
			return (sample['image'].to(self.device), sample['keypoints'].to(self.device)), torch.Tensor([float(sample['label'])])

	def eval_ablation(self, heights, preds, ground_truths):
		"""
			Evaluate using the ablation study
		"""
		preds = preds.detach().cpu().numpy()
		ground_truths = ground_truths.detach().cpu().numpy()

		perc_1 = np.percentile(heights, 5)
		perc_2 = np.percentile(heights, 50)
		perc_3 = np.percentile(heights, 95)



		preds_1 = preds[heights <= perc_1]
		ground_truths_1 = ground_truths[heights <= perc_1]



		preds_2 = preds[(heights > perc_1) & (heights <= perc_2)]
		ground_truths_2 = ground_truths[(heights > perc_1) & (heights <= perc_2)]

		preds_3 = preds[(heights > perc_2) & (heights <= perc_3)]
		ground_truths_3 = ground_truths[(heights > perc_2) & (heights <= perc_3)]

		preds_4 = preds[heights > perc_3]
		ground_truths_4 = ground_truths[heights > perc_3]

		ap_1, ap_2, ap_3, ap_4 = average_precision_score(ground_truths_1, preds_1), average_precision_score(ground_truths_2, preds_2), average_precision_score(ground_truths_3, preds_3), average_precision_score(ground_truths_4, preds_4)



		acc_1, acc_2, acc_3, acc_4 = get_acc_per_distance(ground_truths_1, preds_1), get_acc_per_distance(ground_truths_2, preds_2), get_acc_per_distance(ground_truths_3, preds_3), get_acc_per_distance(ground_truths_4, preds_4)

		#print('Far :{} | Middle : {} | Close : {}'.format(ap_1, ap_2, ap_3))

		distances = [perc_1, perc_2, perc_3]
		#print(distances)
		#exit(0)

		return ap_1, ap_2, ap_3, ap_4, distances
		#return acc_1, acc_2, acc_3, acc_4, distances
	def evaluate(self, model, device, it=1, heights_=False):
		assert self.split in ["test", "val"]
		model = model.eval()
		model = model.to(device)
		if self.type == 'joints':
			"""
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
					acc += le*binary_acc(pred_label.type(torch.float), y_test.type(torch.float).view(-1,1)).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
				acc /= len(new_data)
				ap = average_precision(out_lab, test_lab)
				accs.append(acc)
				aps.append(ap)
			return np.mean(aps), np.mean(accs)"""
			tab_X, tab_Y = self.X.cpu().detach().numpy(), self.Y
			idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
			idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
			positive_samples = np.array(tab_X)[idx_Y1]
			positive_samples_labels = np.array(tab_Y)[idx_Y1]
			N_pos = len(idx_Y1)
			#print(N_pos)
			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			distances = []
			for i in range(it):
				np.random.seed(i)
				np.random.shuffle(idx_Y0)
				neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
				neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]


				total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
				total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()

				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

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
					acc += le*binary_acc(pred_label.type(torch.float), y_test.type(torch.float).view(-1)).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
				if heights_:
					ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)
					distances.append(distance)



				acc /= len(new_data)
				ap = average_precision(out_lab, test_lab)


				accs.append(acc)
				aps.append(ap)
			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			else:
				return np.mean(aps), np.mean(accs)
		elif '+' not in self.type:
			tab_X, tab_Y = self.X, self.Y
			idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
			idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
			positive_samples = np.array(tab_X)[idx_Y1]
			positive_samples_labels = np.array(tab_Y)[idx_Y1]
			N_pos = len(idx_Y1)

			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			distances = []
			for i in range(it):
				np.random.seed(i)
				np.random.shuffle(idx_Y0)
				neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
				neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]

				total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
				total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()
				new_data = Eval_Dataset_crop(self.path_data, total_samples, total_labels, self.type, self.transform)
				data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
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
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
				acc /= len(new_data)
				#rint(torch.round(out_lab).shape)
				#print(test_lab.shape)

				#acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
				ap = average_precision(out_lab, test_lab)
				#print(ap)
				accs.append(acc)
				aps.append(ap)
				if heights_:
					ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)
					distances.append(distance)



				acc /= len(new_data)
				ap = average_precision(out_lab, test_lab)
				#break

			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			return np.mean(aps), np.mean(accs)
		else:
			model = model.eval()
			model = model.to(device)
			#print("Starting evalutation ..")
			tab_X, tab_Y, kps = self.X, self.Y, self.kps.cpu().detach().numpy()
			idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
			idx_Y0 = np.where(np.array(tab_Y) == 0)[0]

			positive_samples = np.array(tab_X)[idx_Y1]
			positive_samples_kps = np.array(kps)[idx_Y1]
			positive_samples_labels = np.array(tab_Y)[idx_Y1]
			N_pos = len(idx_Y1)
			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			distances = []

			for i in range(it):
				np.random.seed(i)
				np.random.shuffle(idx_Y0)

				neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
				neg_samples_kps = np.array(kps)[idx_Y0[:N_pos]]
				neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]

				total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
				total_samples_kps = np.concatenate((positive_samples_kps, neg_samples_kps)).tolist()
				total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()

				new_data = Eval_Dataset_multimodel(self.path_data, total_samples, total_labels, total_samples_kps, self.type, self.transform)
				data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

				acc = 0
				out_lab = torch.Tensor([]).type(torch.float)
				test_lab = torch.Tensor([])
				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
				for x_test, kps_test, y_test in data_loader:
					x_test, kps_test, y_test = x_test.to(device), kps_test.to(device), y_test.to(device)
					output = model((x_test, kps_test))
					out_pred = output
					pred_label = torch.round(out_pred)
					le = x_test.shape[0]
					acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
				acc /= len(new_data)
				#acc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
				ap = average_precision(out_lab, test_lab)
				#print(ap)
				accs.append(acc*100)
				aps.append(ap)
				if heights_:
					ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)
					distances.append(distance)
			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			return np.mean(aps), np.mean(accs)


class Jack_Nu_dataset(Dataset):
	"""
		Class definition for JR and NU dataset (Head / joints / Head+joints)
	"""
	def __init__(self, name, split, type_, path_gt_txt, path_data, pose="full", transform=None, device=torch.device('cpu')):
		"""
		"""
		self.data = None
		self.path_data = path_data
		self.path_gt_txt = path_gt_txt
		self.transform = transform
		self.split = split
		self.pose = pose
		self.type = type_
		self.device = device
		self.name = name
		assert self.name in ['jack', 'nu']
		assert self.type in ['joints', 'heads', 'eyes', 'fullbodies', 'heads+joints', 'eyes+joints', 'fullbodies+joints']
		assert self.pose in ["head", 'full', 'body']
		assert self.split in ['train', 'test', 'val']
		self.file  = open(os.path.join(self.path_gt_txt,'ground_truth_{}.txt'.format(self.name)), "r")

		self.X, self.kps, self.Y = self.preprocess()
	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.Y[idx]

		if self.type == 'joints':
			sample = {'input':self.kps[idx].to(self.device) ,'label':label}
		elif '+' not in self.type:
			sample = {'input': Image.open(os.path.join(self.path_data, self.type + '/' + self.X[idx])), 'label':label}
			if self.transform:
				sample['input'] = self.transform(sample['input']).to(self.device)
		else:
			sample_ = {'image': Image.open(os.path.join(self.path_data, self.type.split('+')[0] + '/' + self.X[idx])), 'keypoints':self.kps[idx] ,'label':label}
			if self.transform:
				sample_['image'] = self.transform(sample_['image'])
			sample = {'input':(sample_['image'].to(self.device), sample_['keypoints'].to(self.device)), 'label':sample_['label']}
		return sample['input'], torch.Tensor([float(sample['label'])])

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
		self.heights = []
		for line in self.file:
			line = line[:-1] # remove the \n
			line_s = line.split(",")
			if line_s[1] == self.split and int(line_s[-1]) != -1: # verify if it's the right split
				X.append(line_s[2]+'.png') # append the image / joints file name
				joints = np.array(json.load(open(os.path.join(self.path_data,line_s[2])))["X"])
				Xs = joints[:17] # take the Xs
				Ys = joints[17:34] # take the Ys
				X_new, Y_new, height = normalize(Xs, Ys, True, True) # normalize according to the paper
				if self.pose == "head":
					X_new, Y_new, C_new = extract_head(X_new, Y_new, joints[34:])
					tensor = np.concatenate((X_new, Y_new, C_new)).tolist()
				elif self.pose == 'body':
					X_new, Y_new, C_new = extract_body(X_new, Y_new, joints[34:])
					tensor = np.concatenate((X_new, Y_new, C_new)).tolist()
				else:
					tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
				kps.append(tensor)
				self.heights.append(height)
				Y.append(int(line_s[-1])) # append the label
		self.file.close()
		return X, torch.tensor(kps), Y

	def eval_ablation(self, heights, preds, ground_truths):
		"""
			Evaluate using the ablation study
		"""
		preds = preds.detach().cpu().numpy()
		ground_truths = ground_truths.detach().cpu().numpy()

		perc_1 = np.percentile(heights, 5)
		perc_2 = np.percentile(heights, 50)
		perc_3 = np.percentile(heights, 95)



		preds_1 = preds[heights <= perc_1]
		ground_truths_1 = ground_truths[heights <= perc_1]



		preds_2 = preds[(heights > perc_1) & (heights <= perc_2)]
		ground_truths_2 = ground_truths[(heights > perc_1) & (heights <= perc_2)]

		preds_3 = preds[(heights > perc_2) & (heights <= perc_3)]
		ground_truths_3 = ground_truths[(heights > perc_2) & (heights <= perc_3)]

		preds_4 = preds[heights > perc_3]
		ground_truths_4 = ground_truths[heights > perc_3]

		ap_1, ap_2, ap_3, ap_4 = average_precision_score(ground_truths_1, preds_1), average_precision_score(ground_truths_2, preds_2), average_precision_score(ground_truths_3, preds_3), average_precision_score(ground_truths_4, preds_4)



		acc_1, acc_2, acc_3, acc_4 = get_acc_per_distance(ground_truths_1, preds_1), get_acc_per_distance(ground_truths_2, preds_2), get_acc_per_distance(ground_truths_3, preds_3), get_acc_per_distance(ground_truths_4, preds_4)

		distances = [perc_1, perc_2, perc_3]

		return ap_1, ap_2, ap_3, ap_4, distances

	def evaluate(self, model, device, it=1, heights_=False):
		assert self.split in ["test", "val"]
		model = model.eval()
		model = model.to(device)
		if self.type == 'joints':

			tab_X, tab_Y = self.kps.cpu().detach().numpy(), self.Y
			idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
			idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
			positive_samples = np.array(tab_X)[idx_Y1]
			positive_samples_labels = np.array(tab_Y)[idx_Y1]
			N_pos = len(idx_Y1)
			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			distances = []
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
				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
				for x_test, y_test in data_loader:
					x_test, y_test = x_test.to(device), y_test.to(device)
					output = model(x_test)
					out_pred = output
					pred_label = torch.round(out_pred)
					le = x_test.shape[0]
					acc += le*binary_acc(pred_label.type(torch.float), y_test.type(torch.float).view(-1,1)).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
				acc /= len(new_data)
				ap = average_precision(out_lab, test_lab)
				accs.append(acc)
				aps.append(ap)
				if heights_:
					ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)
					distances.append(distance)
			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			return np.mean(aps), np.mean(accs)
		elif '+' not in self.type:
			tab_X, tab_Y = self.X, self.Y
			idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
			idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
			positive_samples = np.array(tab_X)[idx_Y1]
			positive_samples_labels = np.array(tab_Y)[idx_Y1]
			N_pos = len(idx_Y1)

			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			distances = []
			for i in range(it):
				np.random.seed(i)
				np.random.shuffle(idx_Y0)
				neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
				neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]

				total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
				total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()
				new_data = Eval_Dataset_crop(self.path_data, total_samples, total_labels, self.type, self.transform)
				data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
				acc = 0
				out_lab = torch.Tensor([]).type(torch.float)
				test_lab = torch.Tensor([])
				for x_test, y_test in data_loader:
					x_test, y_test = x_test.to(device), y_test.to(device)
					output = model(x_test)
					out_pred = output
					pred_label = torch.round(out_pred)

					le = x_test.shape[0]
					acc += le*binary_acc(pred_label.type(torch.float).view(-1), y_test.view(-1,1)).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
				acc /= len(new_data)
				#rint(torch.round(out_lab).shape)
				#print(test_lab.shape)

				#cc = sum(torch.round(out_lab).to(device) == test_lab.to(device))/len(new_data)
				ap = average_precision(out_lab, test_lab)
				#print(ap)
				accs.append(acc)
				aps.append(ap)
				if heights_:
					ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)
					distances.append(distance)
			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			return np.mean(aps), np.mean(accs)
		else:
			model = model.eval()
			model = model.to(device)
			#print("Starting evalutation ..")
			tab_X, tab_Y, kps = self.X, self.Y, self.kps.cpu().detach().numpy()
			idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
			idx_Y0 = np.where(np.array(tab_Y) == 0)[0]

			positive_samples = np.array(tab_X)[idx_Y1]
			positive_samples_kps = np.array(kps)[idx_Y1]
			positive_samples_labels = np.array(tab_Y)[idx_Y1]
			N_pos = len(idx_Y1)
			aps = []
			accs = []
			aps1 = []
			aps2 = []
			aps3 = []
			aps4 = []
			distances = []

			for i in range(it):
				np.random.seed(i)
				np.random.shuffle(idx_Y0)

				neg_samples = np.array(tab_X)[idx_Y0[:N_pos]]
				neg_samples_kps = np.array(kps)[idx_Y0[:N_pos]]
				neg_samples_labels = np.array(tab_Y)[idx_Y0[:N_pos]]

				total_samples = np.concatenate((positive_samples, neg_samples)).tolist()
				total_samples_kps = np.concatenate((positive_samples_kps, neg_samples_kps)).tolist()
				total_labels = np.concatenate((positive_samples_labels, neg_samples_labels)).tolist()

				new_data = Eval_Dataset_multimodel(self.path_data, total_samples, total_labels, total_samples_kps, self.type, self.transform)
				data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)

				acc = 0
				out_lab = torch.Tensor([]).type(torch.float)
				test_lab = torch.Tensor([])
				if heights_:
					heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
				for x_test, kps_test, y_test in data_loader:
					x_test, kps_test, y_test = x_test.to(device), kps_test.to(device), y_test.to(device)
					output = model((x_test, kps_test))
					pred_label = torch.round(output)
					le = x_test.shape[0]
					acc += le*binary_acc(pred_label.type(torch.float).view(-1, 1), y_test.view(-1,1)).item()
					test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
					out_lab = torch.cat((out_lab.detach().cpu(), output.view(-1).detach().cpu()), dim=0)

				acc /= len(new_data)
				#print(out_lab)
				#print(test_lab)
				#exit(0)
				ap = average_precision(out_lab, test_lab)
				#print(ap)
				accs.append(acc)
				aps.append(ap)
				if heights_:
					ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
					aps1.append(ap_1)
					aps2.append(ap_2)
					aps3.append(ap_3)
					aps4.append(ap_4)
					distances.append(distance)
			if heights_:
				return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
			return np.mean(aps), np.mean(accs)
