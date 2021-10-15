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
import cv2

np.random.seed(0)

class JAAD_Dataset(Dataset):
    """
    Wrapper for JAAD Dataset (head, joints, head+joints)
    Args:
        - path_data: str, path where the data is stored
        - type_: str, data type, must be in ['heads', 'joints', 'heads+joints', 'bb+joints', 'bb']
        - split: str, which split to use. Must be in ['train', 'val', 'test']
        - pose: str, which pose type to use. Must be in ['full', 'head', 'body']
        - split_strategy: str, which splitting strategy of the instances to use. Must be in ['instances', 'scenes']. Please refer to the paper for details
        - transform: torchvision.transforms object, normalization and augmentations used for crops.
        - path_txt: str, path to the txt files containing the ground truths
        - device: torch.device object
    """
    def __init__(self, path_data, type_='joints', split='train', pose="full", split_strategy="original", transform=None, path_txt='./splits_jaad', device=torch.device('cpu')):
        """
            First creates each variables and runs the preprocessing function to get the instances.
        """
        self.path_data = path_data
        self.type = type_
        self.split = split
        self.split_strategy = split_strategy
        self.pose = pose
        self.transform = transform
        self.device = device
        self.name = 'jaad'
        self.HEIGHT = 1980
        self.WIDTH = 1280
        assert self.pose in ['full', 'head', 'body']
        assert self.split_strategy in ['instances', 'scenes']
        assert self.type in ['heads', 'joints', 'heads+joints', 'bb+joints', 'bb', 'eyes+joints', 'eyes']
        
        self.txt = open(os.path.join(path_txt, 'jaad_{}_{}.txt'.format(self.split, self.split_strategy)), 'r')
        
        if self.type in ['heads+joints', 'bb+joints', 'eyes+joints']:
            self.X, self.kps, self.Y = self.preprocess()
        else:
            self.X, self.Y = self.preprocess()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        """
            Getter function in order to get the desired items. The returned items will depend on the dataset mode.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.Y[idx]

        if self.type in ['joints', 'bb']:
            sample = {'input':self.X[idx].to(self.device) ,'label':label}
        elif self.type == 'heads':
            sample = {'input': Image.open(os.path.join(self.path_data,self.X[idx])), 'label':label}
            if self.transform:
                sample['input'] = self.transform(sample['input']).to(self.device)
        elif self.type == 'eyes+joints':
            eyes = Image.open(os.path.join(self.path_data,self.X[idx]))
            sample_ = {'image': eyes, 'keypoints':self.kps[idx] ,'label':label}
            if self.transform:
                sample_['image'] = self.transform(sample_['image'])
            sample = {'input':(sample_['image'].to(self.device), sample_['keypoints'].to(self.device)), 'label':sample_['label']}
        elif self.type == 'eyes':
            eyes = Image.open(os.path.join(self.path_data,self.X[idx]))
            sample_ = {'image': eyes,'label':label}
            if self.transform:
                sample_['image'] = self.transform(sample_['image'])
            sample = {'input':sample_['image'].to(self.device), 'label':sample_['label']}
        else:
            sample_ = {'image': Image.open(os.path.join(self.path_data,self.X[idx])), 'keypoints':self.kps[idx] ,'label':label}
            if self.transform:
                sample_['image'] = self.transform(sample_['image'])
            sample = {'input':(sample_['image'].to(self.device), sample_['keypoints'].to(self.device)), 'label':sample_['label']}
        return sample['input'], torch.Tensor([float(sample['label'])])

    def get_joints(self):
        """
            Utility function to get all the instances and their associated ground truth at once
        """
        return self.X, torch.Tensor(self.Y).unsqueeze(1)

    def preprocess(self):
        """
            Utility function to get, accordingly to the dataset mode the desired instances.
            For joints:
                - Loads each instance and its associated ground truth, normalize them and store them in an array
            For heads:
                - Stores each path to the crops and its associated ground truth and store them in an array
            For heads+joints:
                - Loads both normalized joints and paths to each crop image and store them in a different array each
        """
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
                X_new, Y_new, height = normalize_by_image_(X, Y, height_=True)
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
        elif self.type == 'bb':
            tab_Y = []
            kps = []
            for line in self.txt:
                line = line[:-1]
                line_s = line.split(",")
                x1, y1, x2, y2 = float(line_s[2]), float(line_s[3]), float(line_s[4]), float(line_s[5])
                tensor = [x1/self.WIDTH, y1/self.HEIGHT, x2/self.WIDTH, y2/self.HEIGHT]
                height = abs(y2-y1)
                kps.append(tensor)
                tab_Y.append(int(line_s[-1]))
                self.heights.append(height)
            return torch.tensor(kps), tab_Y
        elif self.type == 'heads':
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
                _, _, height = normalize_by_image_(X, Y, height_=True)
                self.heights.append(height)
            return tab_X, tab_Y
        elif self.type == 'eyes':
            tab_X = []
            tab_Y = []
            for line in self.txt:
                line = line[:-1]
                line_s = line.split(",")
                tab_X.append(line_s[-2][:-4]+'_eyes.png')
                joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
                X = joints[:17]
                Y = joints[17:34]
                _, _, height = normalize_by_image_(X, Y, height_=True)
                tab_Y.append(int(line_s[-1]))
                self.heights.append(height)
            return tab_X, tab_Y
        elif self.type == 'heads+joints' or self.type == 'eyes+joints':
            tab_X = []
            tab_Y = []
            kps = []
            for line in self.txt:
                line = line[:-1]
                line_s = line.split(",")
                #tab_X.append(line_s[-2])
                if self.type != 'eyes+joints':
                    tab_X.append(line_s[-2]) # append the image / joints file name
                else:
                    tab_X.append(line_s[-2][:-4]+'_eyes.png')
                joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
                #print(joints)
                X = joints[:17]
                Y = joints[17:34]
                X_new, Y_new, height = normalize_by_image_(X, Y, height_=True)
                tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
                kps.append(tensor) 
                tab_Y.append(int(line_s[-1]))
                self.heights.append(height)
            return tab_X, torch.tensor(kps), tab_Y
        else:
            tab_X = []
            tab_Y = []
            kps = []
            for line in self.txt:
                line = line[:-1]
                line_s = line.split(",")
                tab_X.append(line_s[-2])
                x1, y1, x2, y2 = float(line_s[2]), float(line_s[3]), float(line_s[4]), float(line_s[5])
                tensor = [x1/self.WIDTH, y1/self.HEIGHT, x2/self.WIDTH, y2/self.HEIGHT]
                height = abs(y2-y1)
                kps.append(tensor)
                tab_Y.append(int(line_s[-1]))
                self.heights.append(height)
            return tab_X, torch.tensor(kps), tab_Y

    
    def eval_ablation(self, heights, preds, ground_truths):
        """
            Evaluate according to the ablation study.
            Args:
                - heights: array of heights
                - preds: array of predictions (torch tensor object)
                - ground_truth: array of ground truths (torch tensor object)
            returns 
                - a tuple of aps for each cluster
                - a tuple of accuracies for each cluster
                - an array containing the boundaries of the distances
        """
        preds = preds.detach().cpu().numpy()
        ground_truths = ground_truths.detach().cpu().numpy()

        perc_1 = 110
        perc_2 = 160
        perc_3 = 240

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

        distances = [perc_1, perc_2, perc_3]
        return (ap_1, ap_2, ap_3, ap_4), (ac_1, ac_2, ac_3, ac_4), distances


    def evaluate(self, model, device, it=1, heights_=False):
        """
            Evaluation script that does the same evaluation protocol as described in the paper
            Args :
                - model: torch model to use (looking model)
                - device: torch device
                - it: int, number of iterations for random sampling
                - hegiths_: bool, enables the ablation study on the heights
            Returns :
                -   Computed AP and accuracies + the results of the ablation study on the distances if applicable
        """
        assert self.split in ["test", "val"] # check that we don't evaluate on training set 
        model = model.to(device)
        model.eval()
        heights = None
        if self.type in ['joints', 'bb']:
            tab_X, tab_Y = self.X.cpu().detach().numpy(), self.Y
            idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
            idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
            positive_samples = np.array(tab_X)[idx_Y1] # Take all the positive samples and their ground truths
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

                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

                new_data = Eval_Dataset_joints(total_samples, total_labels, heights)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=True)
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])
                if heights_:
                    heights = []
                with torch.no_grad():
                    for x_test, y_test, height in data_loader:
                        x_test, y_test = x_test.to(device), y_test.to(device)
                        if heights_:
                            heights.extend(height.detach().numpy().tolist())
                        output = model(x_test)
                        out_pred = output
                        pred_label = torch.round(out_pred)
                        le = x_test.shape[0]
                        acc += le*binary_acc(pred_label.type(torch.float), y_test.type(torch.float).view(-1)).item()
                        test_lab = torch.cat((test_lab.detach().cpu(), y_test.type(torch.float).view(-1).detach().cpu()), dim=0)
                        out_lab = torch.cat((out_lab.detach().cpu(), out_pred.view(-1).detach().cpu()), dim=0)
                if heights_:
                    out_ap, out_ac, distance = self.eval_ablation(np.array(heights), out_lab, test_lab)
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
        elif self.type in ['heads', 'eyes']:
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
                new_data = Eval_Dataset_heads(self.path_data, total_samples, total_labels, self.transform)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)

                
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])
                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

                with torch.no_grad():
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
            model.eval()
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
            acs1 = []
            acs2 = []
            acs3 = []
            acs4 = []
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
                
                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

                new_data = Eval_Dataset_heads_joints(self.path_data, total_samples, total_labels, total_samples_kps, self.transform, heights)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)
                
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])

                if heights_:
                    heights = []
                

                with torch.no_grad():
                    for x_test, kps_test, y_test, height in data_loader:
                        x_test, kps_test, y_test = x_test.to(device), kps_test.to(device), y_test.to(device)
                        if heights_:
                            heights.extend(height.detach().numpy().tolist())
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
                if heights_:
                    #ap_1, ap_2, ap_3, ap_4, distance = self.eval_ablation(heights, out_lab, test_lab)
                    out_ap, out_ac, distance = self.eval_ablation(np.array(heights), out_lab, test_lab)
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
            if heights_:
                return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
            return np.mean(aps), np.mean(accs)

class PIE_Dataset(Dataset):
    """Wrapper for PIE Dataset (head, joints, head+joints)"""
    def __init__(self, path_data, type_='joints', split='train', pose="full", split_strategy="scenes", transform=None, path_txt='./splits_jaad', device=torch.device('cpu')):
        self.path_data = path_data
        self.type = type_
        self.split = split
        self.split_strategy = split_strategy
        self.pose = pose
        self.transform = transform
        self.device = device
        self.name = 'pie'
        self.nb = 20000
        assert self.pose in ['full', 'head', 'body']
        assert self.split_strategy in ['instances', 'scenes']
        assert self.type in ['heads', 'joints', 'heads+joints', 'eyes+joints', 'eyes']
        
        self.txt = open(os.path.join(path_txt, 'pie_{}_{}.txt'.format(self.split, self.split_strategy)), 'r')
        
        if self.type in ['heads+joints', 'eyes+joints']:
            self.X, self.kps, self.Y = self.preprocess()
        else:
            self.X, self.Y = self.preprocess()
        
    def balance(self, tab_X):
        indices = np.arange(np.array(tab_X).shape[0])
        np.random.shuffle(indices)
        indices = indices[:self.nb]
        return indices


    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.Y[idx]

        if self.type == 'joints':
            sample = {'input':self.X[idx].to(self.device) ,'label':label}
        elif self.type in ['heads', 'eyes']:
            sample = {'input': Image.open(os.path.join(self.path_data,self.X[idx])), 'label':label}
            if self.transform:
                sample['input'] = self.transform(sample['input']).to(self.device)
        else:
            sample_ = {'image': Image.open(os.path.join(self.path_data,self.X[idx])), 'keypoints':self.kps[idx] ,'label':label}
            if self.transform:
                sample_['image'] = self.transform(sample_['image'])
            sample = {'input':(sample_['image'].to(self.device), sample_['keypoints'].to(self.device)), 'label':sample_['label']}
        return sample['input'], torch.Tensor([float(sample['label'])])

    def get_joints(self):
        return self.X, torch.Tensor(self.Y).unsqueeze(1)

    def preprocess(self):
        self.heights = []
        if self.split == 'test':
            if self.type == 'joints':
                tab_Y = []
                kps = []
                for line in self.txt:
                    line = line[:-1]
                    line_s = line.split(",")
                    joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
                    X = joints[:17]
                    Y = joints[17:34]
                    X_new, Y_new, height = normalize_by_image_(X, Y, height_=True)
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
            elif self.type in ['heads', 'eyes']:
                tab_X = []
                tab_Y = []
                for line in self.txt:
                    line = line[:-1]
                    line_s = line.split(",")
                    if self.type == 'heads':
                        tab_X.append(line_s[-2])
                    else:
                        tab_X.append(line_s[-2][:-4]+'_eyes.png')
                    tab_Y.append(int(line_s[-1]))
                    #joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
                    #X = joints[:17]
                    #Y = joints[17:34]
                    #X_new, Y_new, height = normalize(X, Y, divide=True, height_=True)
                    height = abs(float(line_s[-3])-float(line_s[-5]))
                    self.heights.append(height)
                return tab_X, tab_Y
            else:
                tab_X = []
                tab_Y = []
                kps = []
                for line in self.txt:
                    line = line[:-1]
                    line_s = line.split(",")
                    if self.type != 'eyes+joints':
                        tab_X.append(line_s[-2]) # append the image / joints file name
                    else:
                        tab_X.append(line_s[-2][:-4]+'_eyes.png')
                    joints = np.array(json.load(open(os.path.join(self.path_data, line_s[-2]+'.json')))["X"])
                    #print(joints)
                    X = joints[:17]
                    Y = joints[17:34]
                    X_new, Y_new, height = normalize_by_image_(X, Y, height_=True)
                    tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
                    kps.append(tensor) 
                    tab_Y.append(int(line_s[-1]))
                    self.heights.append(height)
                return tab_X, torch.tensor(kps), tab_Y
        else:
            if self.type == 'joints':
                tab_X = []
                tab_Y = []
                for line in self.txt:
                    line = line[:-1]
                    line_s = line.split(",")
                    tab_X.append(line_s[-2])
                    tab_Y.append(int(line_s[-1]))
                indices = self.balance(tab_X)
                tab_X = np.array(tab_X)[indices]
                tab_Y = np.array(tab_Y)[indices]
                kps = []
                for im in tab_X:
                    joints = np.array(json.load(open(os.path.join(self.path_data, im+'.json')))["X"])
                    X = joints[:17]
                    Y = joints[17:34]
                    X_new, Y_new, height = normalize_by_image_(X, Y, height_=True)
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
                return torch.tensor(kps), tab_Y
            elif self.type == 'heads':
                tab_X = []
                tab_Y = []
                for line in self.txt:
                    line = line[:-1]
                    line_s = line.split(",")
                    tab_X.append(line_s[-2])
                    tab_Y.append(int(line_s[-1]))
                    height = abs(float(line_s[-3])-float(line_s[-5]))
                    self.heights.append(height)
                indices = self.balance(tab_X)
                tab_X = np.array(tab_X)[indices]
                tab_Y = np.array(tab_Y)[indices]
                self.heights = np.array(self.heights)[indices]
                self.heights = self.heights.tolist()
                return tab_X.tolist(), tab_Y.tolist()
            else:
                tab_X = []
                tab_Y = []
                for line in self.txt:
                    line = line[:-1]
                    line_s = line.split(",")
                    #tab_X.append(line_s[-2])
                    if self.type != 'eyes+joints':
                        tab_X.append(line_s[-2]) # append the image / joints file name
                    else:
                        tab_X.append(line_s[-2][:-4]+'_eyes.png')
                    tab_Y.append(int(line_s[-1]))
                indices = self.balance(tab_X)
                tab_X = np.array(tab_X)[indices]
                tab_Y = np.array(tab_Y)[indices]
                kps = []
                for im in tab_X:
                    joints = np.array(json.load(open(os.path.join(self.path_data, im+'.json')))["X"])
                    X = joints[:17]
                    Y = joints[17:34]
                    X_new, Y_new, height = normalize_by_image_(X, Y, height_=True)
                    tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
                    kps.append(tensor)
                    self.heights.append(height)
                #return tab_X.tolist(), torch.tensor(kps), tab_Y.tolist()
            """else:
                tab_X = []
                tab_Y = []
                for line in self.txt:
                    line = line[:-1]
                    line_s = line.split(",")
                    tab_X.append(line_s[-2])
                    tab_Y.append(int(line_s[-1]))
                indices = self.balance(tab_X)
                tab_X = np.array(tab_X)[indices]
                tab_Y = np.array(tab_Y)[indices]
                kps = []
                for im in tab_X:
                    joints = np.array(json.load(open(os.path.join(self.path_data, im+'.json')))["X"])
                    X = joints[:17]
                    Y = joints[17:34]
                    X_new, Y_new, height = normalize_by_image_(X, Y, height_=True)
                    tensor = np.concatenate((X_new, Y_new, joints[34:])).tolist()
                    kps.append(tensor)
                    self.heights.append(height)"""
            return tab_X.tolist(), torch.tensor(kps), tab_Y.tolist()
    
    def eval_ablation(self, heights, preds, ground_truths):
        """
            Evaluate according to the ablation study.
            Args:
                - heights: array of heights
                - preds: array of predictions (torch tensor object)
                - ground_truth: array of ground truths (torch tensor object)
            returns 
                - a tuple of aps for each cluster
                - a tuple of accuracies for each cluster
                - an array containing the boundaries of the distances
        """
        preds = preds.detach().cpu().numpy()
        ground_truths = ground_truths.detach().cpu().numpy()

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
        model = model.to(device)
        model.eval()
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
                else:
                    heights = None

                new_data = Eval_Dataset_joints(total_samples, total_labels, heights)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])
                heights = []
                with torch.no_grad():
                    for x_test, y_test, height in data_loader:
                        x_test, y_test = x_test.to(device), y_test.to(device)
                        if heights_:
                            heights.append(height)
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
        elif self.type in ['heads', 'eyes']:
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
                new_data = Eval_Dataset_heads(self.path_data, total_samples, total_labels, self.transform)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)

                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])
                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

                with torch.no_grad():
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
            model.eval()
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
            acs1 = []
            acs2 = []
            acs3 = []
            acs4 = []
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
                
                new_data = Eval_Dataset_heads_joints(self.path_data, total_samples, total_labels, total_samples_kps, self.transform)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)
                
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])

                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
                with torch.no_grad():
                    for x_test, kps_test, y_test, _ in data_loader:
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
            if heights_:
                return np.mean(aps), np.mean(accs), np.mean(aps1), np.mean(aps2), np.mean(aps3), np.mean(aps4), np.array(distances)
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

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.data_y[idx]
        sample = {'image': self.data_x[idx], 'label':label}
        if self.heights is not None:
            height = self.heights[idx]
        else:
            height = []
        return torch.Tensor(sample['image']), sample['label'], height

class Eval_Dataset_heads(Dataset):
    """JAAD dataset for training and inference"""

    def __init__(self, path_jaad, data_x, data_y, transform=None, heights=None):
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


    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.data_y[idx]
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

class Eval_Dataset_heads_joints(Dataset):
	"""JAAD dataset for training and inference"""

	def __init__(self, path_data, data_x, data_y, kps, transform, heights=None):
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

	def __len__(self):
		return len(self.data_y)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.data_y[idx]
		sample = {'keypoints': self.kps[idx], 'image':Image.open(os.path.join(self.path_data, self.data_x[idx])),'label':label}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
		if self.heights is not None:
			height = self.heights[idx]
		else:
			height = []
		return sample['image'], torch.tensor(sample['keypoints']), sample['label'], height

class LOOK_dataset_(Dataset):
    """
        Class definition for JR and NU dataset (Head / joints / Head+joints)
    """
    def __init__(self, split, type_, path_gt_txt, path_data, pose="full", transform=None, device=torch.device('cpu'), data_name='all'):
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
        self.data_name = data_name
        self.name = 'LOOK'
        assert self.type in ['joints', 'heads', 'heads+joints', 'eyes+joints', 'eyes']
        assert self.pose in ["head", 'full', 'body']
        assert self.split in ['train', 'test', 'val']
        print('Eval on: ', self.data_name)
        self.file  = open(os.path.join(self.path_gt_txt,'ground_truth_look.txt'), "r")
        self.X, self.kps, self.Y = self.preprocess()
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.Y[idx]

        if self.type == 'joints':
            sample = {'input':self.kps[idx].to(self.device) ,'label':label}
        elif self.type in ['heads', 'eyes']:
            sample = {'input': Image.open(os.path.join(self.path_data,self.X[idx])), 'label':label}
            if self.transform:
                sample['input'] = self.transform(sample['input']).to(self.device)
        elif self.type == 'eyes+joints':
            eyes = Image.open(os.path.join(self.path_data,self.X[idx]))
            sample_ = {'image': eyes, 'keypoints':self.kps[idx] ,'label':label}
            if self.transform:
                sample_['image'] = self.transform(sample_['image'])
            sample = {'input':(sample_['image'].to(self.device), sample_['keypoints'].to(self.device)), 'label':sample_['label']}
        else:
            sample_ = {'image': Image.open(os.path.join(self.path_data,self.X[idx])), 'keypoints':self.kps[idx] ,'label':label}
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
            if line_s[2] == self.split and int(line_s[-1]) != -1: # verify if it's the right split
                if self.data_name == 'all':
                    if self.type != 'eyes+joints':
                        if self.type == 'eyes':
                            X.append(line_s[3]+'_eyes.png')
                        else:
                            X.append(line_s[3]+'.png') # append the image / joints file name
                    else:
                        X.append(line_s[3]+'_eyes.png')
                    joints = np.array(json.load(open(line_s[3]))["X"])
                    Xs = joints[:17] # take the Xs
                    Ys = joints[17:34] # take the Ys
                    X_new, Y_new, height = normalize_by_image_(Xs, Ys, height_=True, type_=line_s[1]) # normalize according to the paper
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
                else:
                    
                    if self.data_name == line_s[1]:
                        if self.type != 'eyes+joints':
                            if self.type == 'eyes':
                                X.append(line_s[3]+'_eyes.png')
                            else:
                                X.append(line_s[3]+'.png') # append the image / joints file name
                        else:
                            X.append(line_s[3]+'_eyes.png')
                        joints = np.array(json.load(open(os.path.join(self.path_data,line_s[3])))["X"])
                        Xs = joints[:17] # take the Xs
                        Ys = joints[17:34] # take the Ys
                        X_new, Y_new, height = normalize_by_image_(Xs, Ys, height_=True, type_=line_s[1]) # normalize according to the paper
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
            Evaluate according to the ablation study.
            Args:
                - heights: array of heights
                - preds: array of predictions (torch tensor object)
                - ground_truth: array of ground truths (torch tensor object)
            returns 
                - a tuple of aps for each cluster
                - an array containing the boundaries of the distances
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
        model = model.to(device)
        model.eval()
        heights = None
        if self.type == 'joints':
            
            tab_X, tab_Y = self.kps.cpu().detach().numpy(), self.Y
            idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
            idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
            positive_samples = np.array(tab_X)[idx_Y1]
            positive_samples_labels = np.array(tab_Y)[idx_Y1]
            N_pos = len(idx_Y1)
            print(N_pos)
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
                
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])
                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]

                new_data = Eval_Dataset_joints(total_samples, total_labels, heights)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)

                if heights_:
                    heights = []

                with torch.no_grad():
                    for x_test, y_test, height in data_loader:
                        x_test, y_test = x_test.to(device), y_test.to(device)
                        if heights_:
                            heights.extend(height.detach().numpy().tolist())
                        output = model(x_test)
                        out_pred = output
                        
                        le = x_test.shape[0]
                        #out_pred = torch.zeros([le, 1]).to(device)
                        pred_label = torch.round(out_pred)

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
        elif self.type in ['heads', 'eyes']:
            tab_X, tab_Y = self.X, self.Y
            idx_Y1 = np.where(np.array(tab_Y) == 1)[0]
            idx_Y0 = np.where(np.array(tab_Y) == 0)[0]
            positive_samples = np.array(tab_X)[idx_Y1]
            positive_samples_labels = np.array(tab_Y)[idx_Y1]
            N_pos = len(idx_Y1)
            print(N_pos)
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
                new_data = Eval_Dataset_heads(self.path_data, total_samples, total_labels, self.transform)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)

                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])
                with torch.no_grad():
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
            model = model.to(device)
            model.eval()
            
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
                if heights_:
                    heights = np.array(self.heights)[np.concatenate((idx_Y1, idx_Y0[:N_pos]))]
                new_data = Eval_Dataset_heads_joints(self.path_data, total_samples, total_labels, total_samples_kps, self.transform, heights)
                data_loader = torch.utils.data.DataLoader(new_data, batch_size=16, shuffle=False)
                
                acc = 0
                out_lab = torch.Tensor([]).type(torch.float)
                test_lab = torch.Tensor([])

                if heights_:
                    heights = []
                with torch.no_grad():
                    for x_test, kps_test, y_test, height in data_loader:
                        x_test, kps_test, y_test = x_test.to(device), kps_test.to(device), y_test.to(device)
                        output = model((x_test, kps_test))
                        if heights_:
                            heights.extend(height.detach().numpy().tolist())
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
