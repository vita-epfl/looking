import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import pickle
import seaborn as sns
import json
import numpy as np
import cv2
import random
import scipy.stats as st
from scipy import signal

grads = pickle.load(open("grads.p", "rb")).detach().cpu().numpy()

#print(grads)

y_labels = ['nose', 'left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','nose', 'left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','nose', 'left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']

grads_magnitude = abs(grads)
grads_magnitude = grads_magnitude[:17]+grads_magnitude[17:34]+grads_magnitude[34:]
#print(grads_magnitude[:, -1].shape)


with open('/home/younesbelkada/Travail/Project/example/example.png.predictions.json', 'r') as file:
	data = json.load(file)

def convert(data):
	X = []
	Y = []
	i = 0
	while i < 51:
		X.append(data[i])
		Y.append(data[i+1])
		i += 3
	return X, Y


def gkern(kernlen=21, std=1):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def add_circle(img, X, Y):
	for i in range(len(X)):
		cv2.circle(img, (int(X[i]), int(Y[i])), 2, [255, 255, 255], thickness=1, lineType=8, shift=0)

def add_mask(mask, X, Y, w):
	SIZE_ = 30
	for i in range(len(X)):
		mask[int(Y[i])][int(X[i])] = w[i]/np.max(w)

		x, y = np.meshgrid(np.linspace(-1,1,int(SIZE_*w[i]/np.max(w))), np.linspace(-1,1,int(SIZE_*w[i]/np.max(w)))) 
		dst = np.sqrt(x*x+y*y) 
		sigma = 0.4
		muu = w[i]/np.max(w)
		 #print(gauss)
		step = int(SIZE_*w[i]/np.max(w))//2
		# Calculating Gaussian array 
		gauss = gkern(mask[int(Y[i])-step:int(Y[i])-step+int(SIZE_*w[i]/np.max(w)),int(X[i])-step:int(X[i])-step+int(SIZE_*w[i]/np.max(w))].shape[0], 5*muu)
		print(mask[int(Y[i])-step:int(Y[i])-step+int(SIZE_*w[i]/np.max(w)),int(X[i])-step:int(X[i])-step+int(SIZE_*w[i]/np.max(w))].shape)
		mask[int(Y[i])-step:int(Y[i])-step+int(SIZE_*w[i]/np.max(w)),int(X[i])-step:int(X[i])-step+int(SIZE_*w[i]/np.max(w))] += gauss
		#mask[]
		#exit(0)

	return mask
img = cv2.imread('/home/younesbelkada/Travail/Project/example/example.png')
mask = -1*np.ones((img.shape[0], img.shape[1]))
for i in range(len(data)):
	X, Y = convert(data[i]['keypoints'])
	#break

	#plt.scatter(X, Y)
	#plt.show()
	
	
	mask = add_mask(mask, X, Y, grads_magnitude[:, -1].reshape(-1, 1))

mask = np.uint8(255 * mask)

heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_RAINBOW)
result = cv2.addWeighted(img, 0.6, heatmap, 0.35, 0)
cv2.imshow('',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

add_circle(img, X, Y)

cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#imgplot = plt.imshow(img)
"""
exit(0)



ax = sns.heatmap(grads_magnitude[:, -1].reshape(-1,1), linewidth=0.5, yticklabels=y_labels[:17])
plt.show()

