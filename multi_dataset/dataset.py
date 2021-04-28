import json
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils import normalize
from sklearn.utils import shuffle

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

	def __init__(self, split, pose="full"):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = None
		self.path = '../../data/'
		self.transform = transform
		self.split = split
		self.pose = pose
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
		file = open(self.path+'Kitti/kitti_gt.txt', "r")
		for line in file:
			line = line[:-1]
			line_s = line.split(",")
			if line_s[2] == self.split:
				joints = np.array(json.load(open(self.path+'/Kitti/'+line_s[1]+'.json'))["X"])
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


class JAAD_Dataset_joints_new(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, path, path_jaad, split, pose="full", type_="original"):
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
		self.pose = pose
		assert self.pose in ['full', 'head', 'body']
		assert self.type in ['original', 'video']
		if self.type == "video":
			self.txt = open('../splits/jaad_'+self.split+'_scenes_2k30.txt', "r")
		elif self.type == "original":
			self.txt = open('../splits/jaad_'+self.split+'_original_2k30.txt', "r")
		else:
			print("please select a valid type of split ")
			exit(0)
		self.Y, self.kps = self.preprocess()


	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.Y[idx]
		sample = {'keypoints':self.kps[idx] ,'label':label}
		return sample['keypoints'], sample['label']

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
		for line in self.txt:
			line = line[:-1]
			line_s = line.split(",")
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
		return tab_Y, torch.tensor(kps)

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
			#print(ap)
			accs.append(acc)
			aps.append(ap)
		return np.mean(aps), np.mean(accs)

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
		self.path = "../../data/"
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




"""---------------------------------------"""
class Dataset():
    def __init__(self, joints, normalize):
        with open(joints, 'r') as f:
            dic_jo = json.load(f)
        self.normalize = normalize
        self.inputs_all = dic_jo["X"]
        self.outputs_all = dic_jo["Y"]

        self.inputs_all, self.outputs_all = self.filter()
        self.inputs_all = torch.tensor(self.preprocess())
        self.outputs_all = torch.tensor(self.outputs_all)

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        return inputs, outputs

    def preprocess(self):
        output = []
        for joints in self.inputs_all:
            X = joints[:17]
            Y = joints[17:34]
            #print(X)
            if self.normalize:
              X_new, Y_new = normalize(X, Y, True)
              tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
            #print(X_new)
            else:
              tensor = np.concatenate((X, Y, joints[34:])).tolist()
            #exit(0)
            output.append(tensor)
        return output

    def filter(self):
        X = []
        Y = []
        for x in range(len(self.inputs_all)):
            if self.inputs_all[x] != []:
                X.append(self.inputs_all[x])
                Y.append(self.outputs_all[x])
        return X, Y

    def get_joints(self):
        return self.inputs_all

    def get_outputs(self):
        return self.outputs_all

class LookingDataset():
    """
    Dataloader for JAAD looking dataset
    """

    def __init__(self, joints, phase, sig, normalize, flip_=False, sample_train=False):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        assert(phase in ['train', 'val', 'test'])

        with open(joints, 'r') as f:
            dic_jo = json.load(f)
        self.normalize = normalize
        self.flip_ = flip_
        self.inputs_all = torch.tensor(dic_jo[phase]['X'])
        if self.normalize:
            self.inputs_all = torch.tensor(self.preprocess())
        self.names_all = dic_jo[phase]['names']
        if sig:
            self.outputs_all = torch.tensor([dic_jo[phase]['Y'][i][0] for i in range(len(dic_jo[phase]['Y']))]).to(torch.long)
            self.outputs_all = torch.nn.functional.one_hot(self.outputs_all, 2).to(torch.float)
            if self.flip_:
                f_X, f_Y = self.flip_sig()
                self.inputs_all = torch.tensor(np.concatenate((self.inputs_all.detach().cpu().numpy(), f_X)).tolist())
                self.outputs_all = torch.tensor(np.concatenate((self.outputs_all.detach().cpu().numpy(), f_Y)).tolist())
                #print("je suis là "+phase)
        else:
            self.outputs_all = torch.tensor(dic_jo[phase]['Y'])
            if self.flip_:
                f_X, f_Y = self.flip()
                self.inputs_all = torch.tensor(np.concatenate((self.inputs_all.detach().cpu().numpy(), f_X)).tolist())
                self.outputs_all = torch.tensor(np.concatenate((self.outputs_all.detach().cpu().numpy(), f_Y)).tolist())
                #print("je suis là "+phase)
            if sample_train:
                if phase == 'train':
                    nb_pos = len(self.outputs_all[self.outputs_all == 1.0])
                    mask = self.outputs_all == 0.0
                    #print(nb_pos)
                    idx = np.argwhere(np.asarray(mask))
                    #print(self.inputs_all.shape)
                    idx = idx[:, 0]

                    mask2 = self.outputs_all == 1.0
                    #print(nb_pos)
                    idx2 = np.argwhere(np.asarray(mask2))
                    idx2 = idx2[:, 0]

                    inputs_sampled = self.inputs_all[idx]
                    outputs_sampled = self.outputs_all[idx]
                    inputs_sampled, outputs_sampled = shuffle(inputs_sampled, outputs_sampled)

                    #print("heho")
                    #exit(0)

                    self.outputs_all = torch.tensor(np.concatenate((self.outputs_all[idx2].detach().cpu().numpy(), outputs_sampled[:nb_pos].detach().cpu().numpy())))
                    self.inputs_all = torch.tensor(np.concatenate((self.inputs_all[idx2].detach().cpu().numpy(), inputs_sampled[:nb_pos, :].detach().cpu().numpy())))



        #self.names_all = dic_jo[phase]['names']

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        names = self.names_all[idx]
        return inputs, outputs

    def flip(self):
        flipped_X = []
        flipped_Y = []
        for i in range(len(self.inputs_all)):
            if self.outputs_all[i] == 1.0:
                joints = self.inputs_all[i]
                X = joints[:17]
                #print(X)
                Y = joints[17:]+np.random.randint(3)
                m = np.max(X.detach().cpu().numpy())
                X_new = []
                for x in X:
                    if x != 0 and x != -4:
                        d = abs(m-x.item())
                        X_new.append(x.item()+2*d)
                    else:
                        X_new.append(x.item())
                #print(X_new, Y)
                flipped_X.append(np.concatenate((X_new, Y.detach().cpu().numpy())).tolist())
                flipped_Y.append([1.0])
                self.names_all.append(self.names_all[i])
        #print("heho")
        return flipped_X, flipped_Y

    def flip_sig(self):
        flipped_X = []
        flipped_Y = []
        for i in range(len(self.inputs_all)):
            if self.outputs_all[i][1] == 1.0:
                joints = self.inputs_all[i]
                X = joints[:17]
                #print(X)
                Y = joints[17:]+np.random.randint(3)
                m = np.max(X.detach().cpu().numpy())
                X_new = []
                for x in X:
                    if x != 0 and x != -4:
                        d = abs(m-x.item())
                        X_new.append(x.item()+2*d)
                    else:
                        X_new.append(x.item())
                #print(X_new, Y)
                flipped_X.append(np.concatenate((X_new, Y.detach().cpu().numpy())).tolist())
                flipped_Y.append([0.0, 1.0])
                self.names_all.append(self.names_all[i])
        #print("heho")
        return flipped_X, flipped_Y


    def preprocess(self):
    	output = []
    	for joints in self.inputs_all:
    		X = joints[:17]
    		Y = joints[17:34]
    		#print(X)
    		if self.normalize:
    		  X_new, Y_new = normalize(X, Y, True)
    		  tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
    		#print(X_new)
    		else:
    		  tensor = np.concatenate((X, Y, joints[34:])).tolist()
    		#exit(0)
    		output.append(tensor)
    	return output

    def get_nb_names(self):
        return len(np.unique(self.names_all))


    def get_joints(self):
    	return self.inputs_all

    def get_outputs(self):
    	return self.outputs_all

    def get_names(self):
        return self.names_all


class LookingDataset_balanced():
    """
    Dataloader for JAAD looking dataset
    """

    def __init__(self, joints, ph, sig, normalize, flip_=False, sample_train=True):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        phase = ['train', 'val', 'test']

        with open(joints, 'r') as f:
            dic_jo = json.load(f)
        self.normalize = normalize
        self.flip_ = flip_
        self.sample_train = sample_train
        self.inputs_all = torch.tensor(np.concatenate((dic_jo[phase[0]]['X'], dic_jo[phase[1]]['X'], dic_jo[phase[2]]['X'])))
        self.outputs_all = torch.tensor(np.concatenate((dic_jo[phase[0]]['Y'], dic_jo[phase[1]]['Y'], dic_jo[phase[2]]['Y']))).to(torch.float)
        self.names_all = np.concatenate((dic_jo[phase[0]]['names'], dic_jo[phase[1]]['names'], dic_jo[phase[2]]['names']))
        if self.normalize:
            self.inputs_all = torch.tensor(self.preprocess())

        # shuffle the data

        self.inputs_all, self.outputs_all, self.names_all = shuffle(self.inputs_all, self.outputs_all, self.names_all)

        ## Finding the number of positive samples

        nb_pos = len(self.outputs_all[self.outputs_all == 1.0])

        # masking negative samples

        mask = self.outputs_all == 0.0
        idx = np.argwhere(np.asarray(mask))
        idx = idx[:, 0]

        # the mask is idx

        # masking positive samples

        mask2 = self.outputs_all == 1.0
        idx2 = np.argwhere(np.asarray(mask2))
        idx2 = idx2[:, 0]

        # the mask for positive samples is idx2

        inputs_sampled = self.inputs_all[idx][:nb_pos, :]
        outputs_sampled = self.outputs_all[idx][:nb_pos]
        names_sampled = self.names_all[idx][:nb_pos]
        if self.sample_train:
            self.inputs_sampled_tr = self.inputs_all[idx][nb_pos:int(2*nb_pos), :]
            self.outputs_sampled_tr = self.outputs_all[idx][nb_pos:int(2*nb_pos)]
            self.names_sampled_tr = self.names_all[idx][nb_pos:int(2*nb_pos)]

        # take back nb_pos negative samples

        self.outputs_all = torch.tensor(np.concatenate((self.outputs_all[idx2].detach().cpu().numpy(), outputs_sampled.detach().cpu().numpy())))
        self.inputs_all = torch.tensor(np.concatenate((self.inputs_all[idx2].detach().cpu().numpy(), inputs_sampled.detach().cpu().numpy())))
        self.names_all = np.array(self.names_all[idx2].tolist() + names_sampled.tolist())

        # Concatenate both tensors

        assert sum(self.outputs_all) == self.outputs_all.shape[0]/2 # check that we have as many positive as negatives




        names = np.unique(self.names_all)

        rate_train = int(0.6*len(names))
        rate_val = int(0.1*len(names))
        rate_test = int(0.3*len(names))

        #print(names)
        names_train = names[:rate_train]
        names_val = names[rate_train:rate_train+rate_val]
        names_test = names[rate_train+rate_val:]

        mask_train = [n in names_train for n in self.names_all]
        self.index_train = np.argwhere(np.asarray(mask_train))[:, 0]
        if self.sample_train:
            mask_train_add = [n in names_train for n in self.names_sampled_tr]
            self.index_train_add = np.argwhere(np.asarray(mask_train_add))[:, 0]

        mask_val = [n in names_val for n in self.names_all]
        self.index_val = np.argwhere(np.asarray(mask_val))[:, 0]

        mask_test = [n in names_test for n in self.names_all]
        self.index_test = np.argwhere(np.asarray(mask_test))[:, 0]

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        names = self.names_all[idx]
        return inputs, outputs

    def get_train(self):
        if self.sample_train:
            inputs = torch.tensor(np.concatenate((self.inputs_all[self.index_train].detach().cpu().numpy(), self.inputs_sampled_tr[self.index_train_add].detach().cpu().numpy())))
            outputs = torch.tensor(np.concatenate((self.outputs_all[self.index_train].detach().cpu().numpy(), self.outputs_sampled_tr[self.index_train_add].detach().cpu().numpy())))
            names = np.array(self.names_all[self.index_train].tolist() + self.names_sampled_tr[self.index_train_add].tolist())
            #return self.inputs_all[self.index_train], self.outputs_all[self.index_train], self.names_all[self.index_train]
            return inputs, outputs, names
        else:
            return self.inputs_all[self.index_train], self.outputs_all[self.index_train], self.names_all[self.index_train]


    def get_val(self):
        return self.inputs_all[self.index_val], self.outputs_all[self.index_val], self.names_all[self.index_val]

    def get_test(self):
        return self.inputs_all[self.index_test], self.outputs_all[self.index_test], self.names_all[self.index_test]

    def preprocess(self):
        output = []
        for joints in self.inputs_all:
            X = joints[:17]
            Y = joints[17:34]
            #print(X)
            if self.normalize:
              X_new, Y_new = normalize(X, Y, True)
              tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
            #print(X_new)
            else:
              tensor = np.concatenate((X, Y, joints[34:])).tolist()
            #exit(0)
            output.append(tensor)
        return output

    def get_nb_names(self):
        return len(np.unique(self.names_all))


    def get_joints(self):
        return self.inputs_all

    def get_outputs(self):
        return self.outputs_all
    def get_names(self):
        return self.names_all

class LookingDataset_balanced_random():
    """
    Dataloader for JAAD looking dataset
    """

    def __init__(self, joints, ph, sig, normalize, flip_=False, sample_train=True):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        phase = ['train', 'val', 'test']

        with open(joints, 'r') as f:
            dic_jo = json.load(f)
        self.normalize = normalize
        self.flip_ = flip_
        self.inputs_all = torch.tensor(np.concatenate((dic_jo[phase[0]]['X'], dic_jo[phase[1]]['X'], dic_jo[phase[2]]['X'])))
        self.outputs_all = torch.tensor(np.concatenate((dic_jo[phase[0]]['Y'], dic_jo[phase[1]]['Y'], dic_jo[phase[2]]['Y']))).to(torch.float)
        self.names_all = np.concatenate((dic_jo[phase[0]]['names'], dic_jo[phase[1]]['names'], dic_jo[phase[2]]['names']))
        if self.normalize:
            self.inputs_all = torch.tensor(self.preprocess())

        # shuffle the data

        self.inputs_all, self.outputs_all, self.names_all = shuffle(self.inputs_all, self.outputs_all, self.names_all)

        ## Finding the number of positive samples

        nb_pos = len(self.outputs_all[self.outputs_all == 1.0])

        # masking negative samples

        mask = self.outputs_all == 0.0
        idx = np.argwhere(np.asarray(mask))
        idx = idx[:, 0]

        # the mask is idx

        # masking positive samples

        mask2 = self.outputs_all == 1.0
        idx2 = np.argwhere(np.asarray(mask2))
        idx2 = idx2[:, 0]

        # the mask for positive samples is idx2

        inputs_sampled = self.inputs_all[idx][:nb_pos, :]
        outputs_sampled = self.outputs_all[idx][:nb_pos]
        names_sampled = self.names_all[idx][:nb_pos]

        # take back nb_pos negative samples

        self.outputs_all = torch.tensor(np.concatenate((self.outputs_all[idx2].detach().cpu().numpy(), outputs_sampled.detach().cpu().numpy())))
        self.inputs_all = torch.tensor(np.concatenate((self.inputs_all[idx2].detach().cpu().numpy(), inputs_sampled.detach().cpu().numpy())))
        self.names_all = np.array(self.names_all[idx2].tolist() + names_sampled.tolist())

        # Concatenate both tensors

        assert sum(self.outputs_all) == self.outputs_all.shape[0]/2 # check that we have as many positive as negatives

        self.inputs_all, self.outputs_all, self.names_all = shuffle(self.inputs_all, self.outputs_all, self.names_all)


        names = np.unique(self.names_all)

        rate_train = int(0.6*len(self.inputs_all))
        rate_val = int(0.1*len(self.inputs_all))
        rate_test = int(0.3*len(self.inputs_all))

        self.index_train = rate_train

        self.index_val = rate_val

        self.index_test = rate_test

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        names = self.names_all[idx]
        return inputs, outputs

    def get_train(self):
        return self.inputs_all[:self.index_train], self.outputs_all[:self.index_train], self.names_all[:self.index_train]

    def get_val(self):
        return self.inputs_all[self.index_train:self.index_train+self.index_val], self.outputs_all[self.index_train:self.index_train+self.index_val], self.names_all[self.index_train:self.index_train+self.index_val]

    def get_test(self):
        return self.inputs_all[self.index_train+self.index_val:self.index_train+self.index_val+self.index_test], self.outputs_all[self.index_train+self.index_val:self.index_train+self.index_val+self.index_test], self.names_all[self.index_train+self.index_val:self.index_train+self.index_val+self.index_test]

    def preprocess(self):
        output = []
        for joints in self.inputs_all:
            X = joints[:17]
            Y = joints[17:34]
            #print(X)
            if self.normalize:
              X_new, Y_new = normalize(X, Y, True)
              tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
            #print(X_new)
            else:
              tensor = np.concatenate((X, Y, joints[34:])).tolist()
            #exit(0)
            output.append(tensor)
        return output

    def get_nb_names(self):
        return len(np.unique(self.names_all))


    def get_joints(self):
        return self.inputs_all

    def get_outputs(self):
        return self.outputs_all
    def get_names(self):
        return self.names_all

class PIE_Dataset():
    """
        DataLoader for PIE looking dataset
    """

    def __init__(self, joints):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        np.random.seed(800)

        with open(joints, 'r') as f:
            dic_jo = json.load(f)

        inputs_all = np.array(dic_jo["X"])
        outputs_all = np.array(dic_jo["Y"])
        names_all = np.array(dic_jo["names"])

        #print(outputs_all)

        mask_look = np.where(outputs_all == 1)[0]


        names_look = names_all[mask_look]



        mask_not_look = np.where(outputs_all == 0.0)[0]

        #print(len(np.unique(names_look)))

        mask_not_look_names = ~np.isin(names_all[mask_not_look], names_look)

        #mask_not_look = np.where()[0]


        #print(inputs_all[mask_look])
        inputs_not_look = inputs_all[mask_not_look][mask_not_look_names]
        outputs_not_look = outputs_all[mask_not_look][mask_not_look_names]
        names_not_look = names_all[mask_not_look][mask_not_look_names]
        #print(len(np.unique(names_all[mask_not_look][mask_not_look_names])))


        inputs_not_look, outputs_not_look, names_not_look = shuffle(inputs_not_look, outputs_not_look, names_not_look)

        #print(len(np.unique(names_not_look[:len(mask_look)])))

        #exit(0)

        self.inputs_balanced = np.concatenate((inputs_all[mask_look],inputs_not_look[:len(mask_look)]))
        self.outputs_balanced = np.concatenate((outputs_all[mask_look],outputs_not_look[:len(mask_look)]))

        #print(len(inputs_balanced))
        #exit(0)

        self.inputs_balanced = self.preprocess(True)

        names_all_balanced = np.concatenate((names_all[mask_look],names_not_look[:len(mask_look)]))


        names = shuffle(np.unique(names_all_balanced))


        self.names_all = names_all_balanced

        #print(len(names))

        #exit(0)

        rate_train = int(0.6*len(names))
        rate_val = int(0.1*len(names))
        rate_test = int(0.3*len(names))

        #print(names)
        names_train = names[:rate_train]
        names_val = names[rate_train:rate_train+rate_val]
        names_test = names[rate_train+rate_val:]

        self.mask_train = [n in names_train for n in names_all_balanced]
        #self.index_train = np.argwhere(np.asarray(mask_train))[:, 0]

        self.mask_val = [n in names_val for n in names_all_balanced]
        #self.index_val = np.argwhere(np.asarray(mask_val))[:, 0]

        self.mask_test = [n in names_test for n in names_all_balanced]
        #self.index_test = np.argwhere(np.asarray(mask_test))[:, 0]



    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_balanced.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        names = self.names_all[idx]
        return inputs, outputs

    def get_train(self):
        return torch.tensor(self.inputs_balanced[self.mask_train].tolist()).type(torch.float), torch.tensor(self.outputs_balanced[self.mask_train].tolist()).type(torch.float), self.names_all[self.mask_train]

    def get_val(self):
        return torch.tensor(self.inputs_balanced[self.mask_val].tolist()).type(torch.float), torch.tensor(self.outputs_balanced[self.mask_val].tolist()).type(torch.float), self.names_all[self.mask_val]

    def get_test(self):
        return torch.tensor(self.inputs_balanced[self.mask_test].tolist()).type(torch.float), torch.tensor(self.outputs_balanced[self.mask_test].tolist()).type(torch.float), self.names_all[self.mask_test]


    def preprocess(self, normalizess):
        output = []
        for joints in self.inputs_balanced:
            X = joints[:17]
            Y = joints[17:34]
            #print(X)
            if normalizess:
              X_new, Y_new = normalize(X, Y, True)
              tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
            #print(X_new)
            else:
              tensor = np.concatenate((X, Y, joints[34:])).tolist()
            #exit(0)
            output.append(tensor)
        return np.array(output)

    def get_nb_names(self):
        return len(np.unique(self.names_all))



class Data():
    """
    Dataloader for JAAD looking dataset
    """

    def __init__(self, joints, output):
        self.inputs_all = joints
        self.outputs_all = output
    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        return inputs, outputs


    def get_joints(self):
        return self.inputs_all

    def get_outputs(self):
        return self.outputs_all
