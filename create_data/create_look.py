import os
import json
from glob import glob
import sys
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import errno

parser = argparse.ArgumentParser(description='Creating LOOK dataset')
parser.add_argument('--path_gt', dest='pgt', type=str, help='path to the annotation file', default="./annotations.csv")
parser.add_argument('--path_out_txt', dest='pot', type=str, help='path to the output txt files', default="./splits_look")
parser.add_argument('--path_output_files', dest='pof', type=str, help='path to the input images', default="/data/younes-data/LOOK/LOOK_all")
parser.add_argument('--mode', dest='mo', type=str, help='dataset mode', default="all")
parser.add_argument('--path_keypoints', dest='pkps', type=str, help='path to the pifpaf files', default="/data/younes-data")
parser.add_argument('--path_images', dest='pimg', type=str, help='path to the image files', default="/data/younes-data/LOOK")

args = parser.parse_args()

out_txt = args.pot
anno_file = args.pgt
path_look_keypoints = args.pkps
path_output_files = args.pof
path_images = args.pimg
folders = ['Nuscenes', 'JRDB', 'Kitti']

IOU_THRESHOLD = 0.3

def init_job():
    """
    Initlalize the job by creating 3 txt files corresponding to each subset of the LOOK dataset
    """
    for f in folders:
        directory_out = os.path.join(path_output_files, f)
        try:
            os.makedirs(directory_out)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    return open(os.path.join(out_txt, 'ground_truth_look.txt'), 'w'), open(anno_file, 'r')

def parse_annotation(file):
    """
    Utility function to parse the LOOK annotation file to a dictionnary
    """
    dict_output = {}
    stats_data = {}
    
    for line in file:
        if line.split(',')[0] != 'filename':
            line = line[:-1]
            image_name = line.split(',')[0]
            if image_name not in dict_output:
                dict_output[image_name] = {'bboxes':[], 'splits':[], 'labels':[], 'lines':[]}
            bbox = [float(line.split(',')[1]), float(line.split(',')[2]), float(line.split(',')[3]), float(line.split(',')[4])]
            label = int(line.split(',')[-1])
            split = line.split(',')[5]
            if label != -1:
                dict_output[image_name]['bboxes'].append(bbox)
                dict_output[image_name]['splits'].append(split)
                dict_output[image_name]['labels'].append(label)
                name_data = line.split(',')[0].split('/')[1]
                if name_data not in stats_data:
                    stats_data[name_data] = 1
                else:
                    stats_data[name_data] += 1
    return dict_output, stats_data

def enlarge_bbox(bb, enlarge=1):
    """
    Convert the bounding box from Pifpaf to an enlarged version of it 
    """
    delta_h = (bb[3]) / (7 * enlarge)
    delta_w = (bb[2]) / (3.5 * enlarge)
    bb[0] -= delta_w
    bb[1] -= delta_h
    bb[2] += delta_w
    bb[3] += delta_h
    return bb

def crop(img, bbox):
    """
        Crops the head region given the original image. 
        Args:
            - image : PIL Image object of the input image
            - keypoints : Tensor object of the keypoints
        Returns:
            the cropped eye region
      """
    for i in range(len(bbox)):
        if bbox[i] < 0:
            bbox[i] = 0
        else:
            bbox[i] = int(bbox[i])

    x1, y1, x2, y2 = bbox
    h = y2-y1
    return img[int(y1):int(y1+(h/3)), x1:int(x2)]

def convert_bb(bb):
    """
    Convert the bounding box from a [x, y, w, h] format to a [x1, y1, x2, y2] a format.
    """
    return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]

def crop_eyes(img, keypoints_array_, eps=2):
    """
        Crop the eye region given the crop of the head. 
        Args:
            - image : PIL Image object of the head region
            - keypoints : Tensor object of the keypoints
        Returns:
            the cropped eye region
      """
    keypoints_array = keypoints_array_.copy()
    for i in range(len(keypoints_array)):
        if keypoints_array[i] < 0:
            keypoints_array[i] = 0
    img_width, img_height = img.shape[1], img.shape[0]

    left_most_point = min(keypoints_array[3], keypoints_array[4], keypoints_array[0], keypoints_array[1], keypoints_array[2])
    right_most_point = max(keypoints_array[3], keypoints_array[4], keypoints_array[0], keypoints_array[1], keypoints_array[2])


    top_most_point = min(keypoints_array[21], keypoints_array[20], keypoints_array[17], keypoints_array[18], keypoints_array[19])
    bottom_most_point = max(keypoints_array[21], keypoints_array[20], keypoints_array[17], keypoints_array[18], keypoints_array[19])
      
      
    x1, y1 = int(left_most_point-eps), int(top_most_point-eps)
    x2, y2 = int(right_most_point+eps), int(bottom_most_point+eps)

      #print(keypoints[18]-keypoints[19])

    if x1 < 0:
        x1 = 0
    if x2 <= 0:
        x2 = 5
    if y1 < 0:
        y1 = 0
    if y2 <= 0:
        y2 = 5

    if x1 >= img_width:
        x1 = img_width-5
    if x2 >= img_width:
        x2 = img_width
    if y1 >= img_height:
        y1 = img_height-5
    if y2 >= img_height:
        y2 = img_height

    return img[y1:y2, x1:x2]

def bb_intersection_over_union(boxA, boxB):
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

def convert(data):
    """
    Function to convert the keypoints from Pifpaf to our format
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
    return A

def get_indice_max_iou(di_pifpaf, array_bboxes):
    """
    A function to get the index of the maximum IoU value given the output from Pifpaf and the array of bounding
    boxes from the ground truth annotations
    """
    bbox_pifpaf = convert_bb(di_pifpaf['bbox'])
    if len(array_bboxes) != 0:
        ious = np.array([bb_intersection_over_union(bbox_pifpaf, b) for b in array_bboxes])
        if np.max(ious) >= IOU_THRESHOLD:
            return np.where(ious == np.max(ious))[0][0]
    return -1

txt_final_file, file = init_job()
di_annotations, stats_data = parse_annotation(file)


def main():
    """

    """
    i = 0
    counts_data = {}
    for j, keys in tqdm(enumerate(di_annotations)):
        name_data = keys.split('/')[1]
        path_keypoints = keys.replace('LOOK', 'LOOK_keypoints')+'.predictions.json'
        file_keypoints = open(os.path.join(path_look_keypoints, path_keypoints), 'r')
        pifpaf_di = json.load(file_keypoints)

        bboxes_anno = di_annotations[keys]['bboxes']
        looking_labels = di_annotations[keys]['labels']
        splits = di_annotations[keys]['splits']

        img = Image.open(os.path.join(path_images, keys))

        for pifpaf_instances in pifpaf_di:
            index_max_iou = get_indice_max_iou(pifpaf_instances, bboxes_anno)

            if index_max_iou != -1:
                label = looking_labels[index_max_iou]
                if label != -1:
                    out_name = os.path.join(path_output_files, name_data, str(i).zfill(10)+'.json')
                    bbox = bboxes_anno[index_max_iou]
                    if name_data not in counts_data:
                        counts_data[name_data] = 1
                    else:
                        counts_data[name_data] += 1


                    if splits[index_max_iou] == 'train':
                        if counts_data[name_data] >= 0.9*stats_data[name_data]:
                            split = 'val'
                        else:
                            split = 'train'
                    else:
                        split = splits[index_max_iou]

                    bbox_final_without_enlarge = convert_bb(pifpaf_instances['bbox'])
                    converted_kps = convert(pifpaf_instances['keypoints'])
                    di = {"X":converted_kps}
                    json.dump(di, open(out_name, "w"))
					
                    head = crop(np.asarray(img), convert_bb(enlarge_bbox(pifpaf_instances['bbox'])))
                    Image.fromarray(head).convert('RGB').save(os.path.join(path_output_files, out_name+'.png'))

                    eyes = crop_eyes(np.asarray(img), converted_kps)
                    Image.fromarray(eyes).convert('RGB').save(os.path.join(path_output_files,out_name+'_eyes.png'))

                    line = ','.join([os.path.join(path_images, keys), name_data, split, out_name, str(bbox_final_without_enlarge[0]), str(bbox_final_without_enlarge[1]), str(bbox_final_without_enlarge[2]),str(bbox_final_without_enlarge[-1]), str(label)+'\n'])
                    txt_final_file.write(line)
                    i += 1

main()