from datetime import datetime
import cv2
import openpifpaf
import openpifpaf.datasets as datasets
import os
import numpy as np
from openpifpaf.predict import processor_factory, preprocess_factory
from openpifpaf import decoder, network, visualizer, show, logger

"""COCO_PERSON_SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]]
"""
COCO_PERSON_SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]
COCO_HEAD = [
    [3,4]
]
def convert(data):
    """X = []
    Y = []
    C = []
    A = []
    i = 0
    print(data)
    while i < 51:
        X.append(data[i])
        Y.append(data[i+1])
        C.append(data[i+2])
        i += 3
    """
    X, Y, C = data[0], data[1], data[2]
    A = np.array([X, Y, C]).flatten().tolist()
    return X, Y, C, A

def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]


def preprocess_pifpaf(annotations, im_size=None, enlarge_boxes=True, min_conf=0.):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 1 if enlarge_boxes else 2  # Avoid enlarge boxes for social distancing

    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        if conf >= min_conf:
            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)

    return boxes, keypoints

def filecreation(dirname):
    mydir = os.path.join(dirname,'_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

def draw_skeleton(img, kps, color, skeleton=COCO_PERSON_SKELETON):
    X, Y, C, _ = convert(kps)
    linewidth = 4
    height = abs(Y[0]-Y[-1])
    for ci, connection in enumerate(np.array(skeleton) - 1):
        #print(connection)
        c1, c2 = connection
        #print(X[c1])
        img = cv2.line(img,(int(X[c2]),int(Y[c2])),(int(X[c1]),int(Y[c1])), color,linewidth)
        #exit(0)
        """
        c = matplotlib.cm.get_cmap('tab20')(ci / len(self.skeleton))
                if np.all(v[connection] > self.dashed_threshold):
                    ax.plot(x[connection], y[connection],
                            linewidth=self.linewidth, color=c,
                            linestyle='dashed', dash_capstyle='round')
                if np.all(v[connection] > self.solid_threshold):
                    ax.plot(x[connection], y[connection],
                            linewidth=self.linewidth, color=c, solid_capstyle='round')"""
    head = COCO_HEAD[0]
    c1, c2 = head
    #radius = abs(int(X[c1])- int(X[c2]))
    radius = int(0.09*height)
    center = int((X[c1]+X[c2])/2), int((Y[c1]+Y[c2])/2)
    img = cv2.circle(img, center, radius, color, -1)
    img = cv2.circle(img, center, radius, (255, 255, 255), 2)
    return img
def run_and_kps(img, kps, label):
    blk = np.zeros(img.shape, np.uint8)
    X, Y, C, _ = convert(kps)
    if label > 0.5:
        color = (0, 255, 0)
    else:
        color = (0,0,255)
    for i in range(len(Y)):
        blk =cv2.circle(blk, (int(X[i]), int(Y[i])), 1, color, 2)
    #img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
    return blk

def load_pifpaf(args):
    args.figure_width = 10
    args.dpi_factor = 1.0
    args.keypoint_threshold_rel = 0.0
    args.instance_threshold = 0.2
    args.keypoint_threshold = 0
    args.force_complete_pose = True

    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    processor, net = processor_factory(args)
    preprocess = preprocess_factory(args)
    return net, processor, preprocess