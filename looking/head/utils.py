import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.transforms.functional import pad
import numpy as np
import numbers
import torchvision.transforms.functional as F
from math import floor
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

def renormalize(tensor):
	minFrom= tensor.min()
	maxFrom= tensor.max()
	minTo = 0
	maxTo=1
	return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


def crop(img, bbox):
	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])

	x1, y1, w, h = bbox
	return img[int(0.9*y1):int(y1+h), x1:int(x1+w)]


def crop_jaad(img, bbox):

	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])

	x1, y1, x2, y2 = bbox
	return img[int(y1):int(y2), x1:int(x2)]

def crop_jaad2(img, bbox):

	for i in range(len(bbox)):
		if bbox[i] < 0:
			bbox[i] = 0
		else:
			bbox[i] = int(bbox[i])

	x1, y1, x2, y2 = bbox
	#print(y1, y2)
	h = y2-y1
	return img[int(y1):int(y1+(h/3)), x1:int(x2)]

def show(img):
	plt.figure()
	plt.imshow(img)
	plt.show()

def average_precision(output, target):
    return average_precision_score(target.detach().cpu().numpy(), output.detach().cpu().numpy())

def binary_acc(y_pred, y_test):
    #print(y_pred)
    y_pred_tag = torch.round(y_pred)
    #print(y_pred_tag)
    #print(y_test)
    #print(confusion_matrix(y_test.cpu().detach().numpy(), y_pred_tag.cpu().detach().numpy()))
    correct_results_sum = (y_pred_tag == y_test.view(-1)).sum().float()
    #print(correct_results_sum)
    #print(y_test.shape[0])
    acc = correct_results_sum/y_test.shape[0]
    #print(acc)
    acc = acc * 100
    return acc


def normalize(X, Y, divide=True):
    center_p = (int((X[11]+X[12])/2), int((Y[11]+Y[12])/2))
    #X_new = np.array(X)-center_p[0]
    X_new = np.array(X)
    Y_new = np.array(Y)-center_p[1]
    width = abs(np.max(X_new)-np.min(X_new))
    height = abs(np.max(Y_new)-np.min(Y_new))
    #print(X_new)
    if divide:
        Y_new /= max(width, height)
        X_new /= max(width, height)
        #Y_new /= max(abs(np.min(Y_new)), abs(np.max(Y_new)))
        #X_new /= max(abs(np.min(X_new)), abs(np.max(X_new)))
    #print(X_new)
    #exit(0)
    return X_new, Y_new