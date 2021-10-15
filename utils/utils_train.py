
import numpy as np
from math import floor
import torch
import cv2
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import json

### Visualisation functions

def convert_bb(bb):
    """
        Convert the bounding box format from:
            x1, y1, w, h to x1, y1, x2, y2
    """
    return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]

def bb_intersection_over_union(boxA, boxB):
    """
        Code for computing the IoU given 2 converted boxes
    """     
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=10):
    """
        Utility function for drawing lines
    """
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            img = cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                img = cv2.line(img,s,e,color,thickness)
            i+=1
    return img

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    """
        Utility function for drawing polygones
    """
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        img = drawline(img,s,e,color,thickness,style)
    return img

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    """
        Utility function for drawing rectangles
    """
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    return drawpoly(img,pts,color,thickness,style)

def convert(data):
    """
        Utility function to convert the data format of Pifpaf to our convention.
        Args:
            - joints with Pifpaf's format
        Returns:
            - 4 arrays, X coordinates, Y coordinates, C confidences scores and A the concatenated arrays
    """
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

def run_and_rectangle(img, data, model, device):

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale             = 0.30
    fontColor             = (0,0,0)
    lineType               = 1

    blk = np.zeros(img.shape, np.uint8)
    inputs = []
    bboxes = []
    enlarge = 1
    for i in range(len(data)):
        X, Y, C, A = convert(data[i]['keypoints'])
        bb = data[i]['bbox']
        bboxes.append(bb)

        #print(bb)
        #inp[:17], inp[17:34], inp[34:]
        delta_h = (bb[3]) / (7 * enlarge)
        delta_w = (bb[2]) / (3.5 * enlarge)
            #assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        bb[0] -= delta_w
        bb[1] -= delta_h
        bb[2] += delta_w
        bb[3] += delta_h
        X_new, Y_new = normalize(X, Y)
        inp = torch.tensor(np.concatenate((X_new, Y_new, C)).tolist()).to(device).view(1, -1)
        #print(inp.shape)
        pred = model(inp).item()
        inputs.append(A)
        #break
        if pred >= 0.5:
            blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,255,0), 1)
            blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1]+bb[3])-10), (int(bb[0]+30), int(bb[1]+bb[3])), (0,255,0), -1)
            #look.append(1)
        else:
            blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,0,255), 1)
            blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1]+bb[3])-10), (int(bb[0]+30), int(bb[1]+bb[3])), (0,0,255), -1)
            #look.append(0)
        cv2.putText(blk,str("%.2f" % pred), (int(bb[0])+4, int(bb[1]+bb[3])-3), font,   fontScale,fontColor,lineType)

            #break
    img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
    return img

def run_and_rectangle_saved(img, data, data_gt, model, device):

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale             = 0.30
    fontColor             = (0,0,0)
    lineType               = 1

    blk = np.zeros(img.shape, np.uint8)
    inputs = []
    bboxes = []
    enlarge = 1
    bboxes_gt = data_gt["bbox"]
    for i in range(len(data)):
        X, Y, C, A = convert(data[i]['keypoints'])
        bb = data[i]['bbox']
        bboxes.append(bb)
        ious = np.array([bb_intersection_over_union(convert_bb(bb), convert_bb(b)) for b in bboxes_gt])
        #print(ious)
        index_gt = np.where(ious == np.max(ious))[0][0]
        #print(np.where(ious == np.max(ious)))
        #print(bb)
        #inp[:17], inp[17:34], inp[34:]
        delta_h = (bb[3]) / (7 * enlarge)
        delta_w = (bb[2]) / (3.5 * enlarge)
            #assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        bb[0] -= delta_w
        bb[1] -= delta_h
        bb[2] += delta_w
        bb[3] += delta_h
        X_new, Y_new = normalize(X, Y)
        inp = torch.tensor(np.concatenate((X_new, Y_new, C)).tolist()).to(device).view(1, -1)
        #print(inp.shape)
        pred = model(inp).item()
        inputs.append(A)
        #break
        label = data_gt["Y"][index_gt]
        if pred >= 0.5:
            if round(pred) == label:
                blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,255,0), 1)
                blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1]+bb[3])-10), (int(bb[0]+30), int(bb[1]+bb[3])), (0,255,0), -1)
            else:
                blk = drawrect(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,255,0), 1)
                blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1]+bb[3])-10), (int(bb[0]+30), int(bb[1]+bb[3])), (0,255,0), -1)
            #look.append(1)
        else:
            if round(pred) == label:
                blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,0,255), 1)
                blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1]+bb[3])-10), (int(bb[0]+30), int(bb[1]+bb[3])), (0,0,255), -1)
            else:
                blk = drawrect(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,0,255), 1)
                blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1]+bb[3])-10), (int(bb[0]+30), int(bb[1]+bb[3])), (0,0,255), -1)
            #look.append(0)
        cv2.putText(blk,str("%.2f" % pred), (int(bb[0])+4, int(bb[1]+bb[3])-3), font,   fontScale,fontColor,lineType)

            #break
    img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
    return img

def run_and_kps(img, data):
    blk = np.zeros(img.shape, np.uint8)
    for di in data:
        kps = di["keypoints"]
        X, Y, C, _ = convert(kps)
        for i in range(len(Y)):
            c = int(255*C[i])
            blk =cv2.circle(blk, (int(X[i]), int(Y[i])), 1, (255, 255-c, 255-c), 2)
    img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
    return img

def extract_head(X, Y, C):
    """
        utility function to extract the head keypoints given the arrays of Xs, Ys, and C Confidence scores
        Please refer to : https://openpifpaf.github.io/datasets.html to understand the order of the keypoints
        Returns:
            - X_new : Array of X positions of the head keypoints
            - Y_new : Array of Y positions of the head keypoints
            - C_new : Array of the confidence scores of the head keypoints
    """
    X_new = X[:5]
    Y_new = Y[:5]
    C_new = C[:5]
    return np.array(X_new), np.array(Y_new), np.array(C_new)

def extract_body(X, Y, C):
    """
        utility function to extract the body keypoints given the arrays of Xs, Ys, and C Confidence scores
        Please refer to : https://openpifpaf.github.io/datasets.html to understand the order of the keypoints
        Returns:
            - X_new : Array of X positions of the body keypoints
            - Y_new : Array of Y positions of the body keypoints
            - C_new : Array of the confidence scores of the body keypoints
    """
    X_new = X[5:]
    Y_new = Y[5:]
    C_new = C[5:]
    return np.array(X_new), np.array(Y_new), np.array(C_new)


### Metrics

def binary_acc(y_pred, y_test):
    """
        Utility function to get the accuracy given the predictions and the ground truths
        Args:
            - y_pred : predicted labels
            - y_test : ground truth labels
        Returns:
            Categorical accurcacy
    """
    y_pred_tag = torch.round(torch.flatten(y_pred))
    y_test = torch.flatten(y_test)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = acc * 100
    return acc


def acc_per_class(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    # matrix = confusion_matrix(y_test.cpu().detach().numpy(), y_pred_tag.cpu().detach().numpy())
    tn, fp, fn, tp = confusion_matrix(y_test.cpu().detach().numpy(), y_pred_tag.cpu().detach().numpy()).ravel()
    acc0 = tn / (tn + fn)
    acc1 = tp / (tp + fp)
    rec1 = tp / (tp + fn)
    rec0 = tn / (tn + fp)
    return acc0, acc1


def acc_rec_per_class(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    # matrix = confusion_matrix(y_test.cpu().detach().numpy(), y_pred_tag.cpu().detach().numpy())
    tn, fp, fn, tp = confusion_matrix(y_test.cpu().detach().numpy(), y_pred_tag.cpu().detach().numpy()).ravel()
    acc0 = tn / (tn + fn)
    acc1 = tp / (tp + fp)
    rec1 = tp / (tp + fn)
    rec0 = tn / (tn + fp)
    return acc0, acc1, rec0, rec1


def save_results(y_pred):
    y_pred_tag = torch.round(y_pred)
    di = {}
    di['output'] = np.asarray(y_pred_tag.cpu().detach().numpy()).tolist()
    with open('output_train.json', 'w') as out_file:
        json.dump(di, out_file)

def get_acc_per_distance(ground_truths_1, pred_1):
    idx_1 = (ground_truths_1==1)
    return sum(np.round(pred_1) == ground_truths_1)/len(ground_truths_1)


def normalize(X, Y, divide=True, height_=False):
    """
        Old normalization techniques, deprecated 
    """
    center_p = (int((X[11] + X[12]) / 2), int((Y[11] + Y[12]) / 2))
    X_new = np.array(X)
    Y_new = np.array(Y) - center_p[1]
    width = abs(np.max(X_new) - np.min(X_new))
    height = abs(np.max(Y_new) - np.min(Y_new))

    if divide==True:
        Y_new /= max(width, height)
        X_new /= max(width, height)
    
    if height_==True:
       return X_new, Y_new, height

    return X_new, Y_new

def normalize_by_image_(X, Y, height_=False, type_='JAAD'):
    """
        Normalize the image according to the paper.
        Args:
            - X: array of X positions of the keypoints
            - Y: array of Y positions of the keypoints
            - height_ : True if you want to return an array of heights
            - type_ : str that describes the dataset type
        Returns:
            if height_ is enabled, returns the normalized arrays and an array of heights. else returns the normalized arrays
    """
    
    if type_ == 'JAAD':
        image_width, image_height = 1980, 1280
    elif type_ == 'Kitti':
        image_width, image_height = 1238, 374
    elif type_ == 'JDRB':
        image_width, image_height = 752, 480
    elif type_ == 'Nuscenes':
        image_width, image_height = 1600, 900
    else:
        image_width, image_height = None, None
    
    center_p = (int((X[11] + X[12]) / 2), int((Y[11] + Y[12]) / 2))
    X_new = np.array(X)/image_width
    Y_new = np.array(Y)-center_p[1]


    width = abs(np.max(X) - np.min(X))
    height = abs(np.max(Y) - np.min(Y))

    X_new = X_new + ((np.array(X)-center_p[0])/width)
    Y_new /= height
    
    if height_==True:
       return X_new, Y_new, height

    return X_new, Y_new


def val_kitti(output, labels):
    output = output.detach().cpu().numpy()
    mu = output[:, 0]
    si = abs(output[:, 1])
    labels = labels.detach().cpu().numpy()
    return np.mean(abs(labels - mu) / si)


def average_precision(output, target):
    """
        Utility function to compute the AP score of 2 torch tensors.
        Args:
            - output : the predicted labels
            - target : the ground truth labels
    """
    return average_precision_score(target.detach().cpu().numpy(), output.detach().cpu().numpy())

def crop_eyes(image, keypoints):
    """
        Crop the eye region given the crop of the head. 
        Args:
            - image : PIL Image object of the head region
            - keypoints : Tensor object of the keypoints
        Returns:
            the cropped eye region
    """
    keypoints_array = keypoints.detach().numpy().tolist()
    width_eyes = abs(keypoints[3]-keypoints[4]) # right ear - left ear
    center_eye_y = int((keypoints[18]-keypoints[19])/2)
    height_eyes = abs(keypoints[17]-center_eye_y) # nose - eye
    x1, y1 = min(keypoints[3], keypoints[4]) ,center_eye_y-int(height_eyes/2)
    print(keypoints[18]-keypoints[19])
    print(keypoints)
    exit(0)

def print_summary(i, EPOCHS, train_loss, acc, acc_val, ap, val_ap, acc1, acc0):
    text = "epoch {}/{} [".format(i, EPOCHS) + int(i * 10 / EPOCHS) * "=" + ">" + floor(
        10 - int(i * 10 / EPOCHS) - 1) * "." + \
           "] train_loss : %.3f | train_acc : %.1f | val_acc: %.1f | train_ap : %0.1f | val_ap : %.1f | " \
           "acc1 : %0.1f | acc0 : %0.1f" % (train_loss, acc, acc_val, ap*100, val_ap*100, acc1*100, acc0*100)
    print('{}'.format(text), end="\r")

def print_summary_step(step, train_loss, acc):
    text = "Step : {} | ".format(step) + "m_step_loss : %.3f | m_step_acc : %.1f " % (train_loss, acc)
    print('{}'.format(text), end="\r")


def parse_str(text):
    text_s = text.split(",")
    return text_s
