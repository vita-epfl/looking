import torch
import numpy as np
import json
from math import floor
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = acc
    return acc

def acc_per_class(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test.cpu().detach().numpy(), y_pred_tag.cpu().detach().numpy()).ravel()
    acc0 = tn/(tn+fn)
    acc1 = tp/(tp+fp)
    rec1 = tp/(tp+fn)
    rec0 = tn/(tn+fp)
    return acc0, acc1

def acc_rec_per_class(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test.cpu().detach().numpy(), y_pred_tag.cpu().detach().numpy()).ravel()
    acc0 = tn/(tn+fn)
    acc1 = tp/(tp+fp)
    rec1 = tp/(tp+fn)
    rec0 = tn/(tn+fp)
    return acc0, acc1, rec0, rec1

def save_results(y_pred):
    y_pred_tag = torch.round(y_pred)
    di = {}
    di['output'] = np.asarray(y_pred_tag.cpu().detach().numpy()).tolist()
    with open('output_train.json', 'w') as out_file:
        json.dump(di, out_file)

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

def val_kitti(output, labels):
    output = output.detach().cpu().numpy()
    mu = output[:, 0]
    si = abs(output[:, 1])
    labels = labels.detach().cpu().numpy()
    return np.mean(abs(labels-mu)/si)
def average_precision(output, target):
    return average_precision_score(target.detach().cpu().numpy(), output.detach().cpu().numpy())

def print_summary(i, EPOCHS, train_loss, acc, acc_val, ap, val_ap, acc1, acc0):
    text = "epoch {}/{} [".format(i, EPOCHS)+int(i*10/EPOCHS)*"="+">"+floor(10-int(i*10/EPOCHS)-1)*"."+"] train_loss : %.5f | train_acc : %0.3f | val_acc: %.3f | train_ap : %0.5f | val_ap : %0.5f | acc1 : %0.5f | acc0 : %0.5f"%(train_loss, acc, acc_val, ap, val_ap, acc1, acc0)
    print('{}'.format(text),  end="\r")

def parse_str(text):
    text_s = text.split(",")
    return text_s
