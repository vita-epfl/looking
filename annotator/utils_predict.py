import torch
import pickle
import json
import numpy as np
import cv2

def convert(data):
	X = []
	Y = []
	C = []
	A = []
	i = 0
	while i < 51:
		X.append(data[i])
		Y.append(data[i+1])
		C.append(data[i+2])
		i += 3
	A = np.array([X, Y, C]).flatten().tolist()
	return X, Y, C, A

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

def run_and_rectangle(img, data, model, device):

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 0.30
	fontColor              = (0,0,0)
	lineType               = 1

	blk = np.zeros(img.shape, np.uint8)
	look = []
	inputs = []
	bboxes = []
	for i in range(len(data)):
		X, Y, C, A = convert(data[i]['keypoints'])
		bb = data[i]['bbox']
		bboxes.append(bb)
		#print(bb)
		#inp[:17], inp[17:34], inp[34:]
		X_new, Y_new = normalize(X, Y)
		inp = torch.tensor(np.concatenate((X_new, Y_new, C)).tolist()).to(device).view(1, -1)
		pred = model(inp).item()
		inputs.append(A)
		#break
		if pred > 0.5:
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,255,0), 1)
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])-10), (int(bb[0]+30), int(bb[1])), (0,255,0), -1)
			look.append(1)
		else:
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,0,255), 1)
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])-10), (int(bb[0]+30), int(bb[1])), (0,0,255), -1)
			look.append(0)
		cv2.putText(blk,str("%.2f" % pred), (int(bb[0])+4, int(bb[1])-3), font,	fontScale,fontColor,lineType)
			
			#break
	img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
	return img, look, inputs, bboxes

def run_and_rectangle_saved(img, data, model, device, data2):
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 0.30
	fontColor              = (0,0,0)
	lineType               = 1

	blk = np.zeros(img.shape, np.uint8)
	look = []
	inputs = []
	bboxes = []
	for i in range(len(data2['Y'])):
		#, Y, C, A = convert(data[i]['keypoints'])
		bb = data2['bbox'][i]
		inputs.append(data2["X"][i])
		bboxes.append(bb)
		#print(bb)
		#inp[:17], inp[17:34], inp[34:]
		#X_new, Y_new = normalize(X, Y)
		#inp = torch.tensor(np.concatenate((X_new, Y_new, C)).tolist()).to(device).view(1, -1)
		pred = data2["Y"][i]
		#inputs.append(A)
		#break
		if pred > 0.5:
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,255,0), 1)
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])-10), (int(bb[0]+30), int(bb[1])), (0,255,0), -1)
			look.append(1)
		else:
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,0,255), 1)
			blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])-10), (int(bb[0]+30), int(bb[1])), (0,0,255), -1)
			look.append(0)
		cv2.putText(blk,str("%.2f" % pred), (int(bb[0])+4, int(bb[1])-3), font,	fontScale,fontColor,lineType)
			
			#break
	img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
	return img, look, inputs, bboxes



def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    #print(rect)
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False



"""
for im in tab_im:
	if im != 'looking':
		#im = 'example.png'
		img = cv2.imread('/home/younesbelkada/Travail/Project/test_set/video_0313/'+im)

		try:
			with open('/home/younesbelkada/Travail/Project/test_set/out_0313/{}.predictions.json'.format(im), 'r') as file:
				data = json.load(file)
			print(data)


			#img = cv2.imread('/home/younesbelkada/Travail/Project/example/example.png')


			font                   = cv2.FONT_HERSHEY_SIMPLEX
			fontScale              = 0.30
			fontColor              = (0,0,0)
			lineType               = 1

			blk = np.zeros(img.shape, np.uint8)
			for i in range(len(data)):
				X, Y, C = convert(data[i]['keypoints'])
				bb = data[i]['bbox']
				#print(bb)
				#inp[:17], inp[17:34], inp[34:]
				X_new, Y_new = normalize(X, Y)
				inp = torch.tensor(np.concatenate((X_new, Y_new, C)).tolist()).to('cuda').view(1, -1)
				pred = model(inp).item()
				#break
				
				if pred > 0.5:
					blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,255,0), 1)
					blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])-10), (int(bb[0]+30), int(bb[1])), (0,255,0), -1)
				else:
					blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,0,255), 1)
					blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])-10), (int(bb[0]+30), int(bb[1])), (0,0,255), -1)

				cv2.putText(blk,str("%.2f" % pred), 
				    (int(bb[0])+4, int(bb[1])-3), 
				    font, 
				    fontScale,
				    fontColor,
				    lineType)
			
			#break
			img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
			#cv2.imshow('',img)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			cv2.imwrite('/home/younesbelkada/Travail/Project/test_set/looking_0313/'+im, img)
		except:
			print(im)
		#break"""