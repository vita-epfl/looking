from datetime import datetime
import cv2
import openpifpaf
import openpifpaf.datasets as datasets
import os
from openpifpaf.predict import processor_factory, preprocess_factory
from openpifpaf import decoder, network, visualizer, show, logger


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

def load_pifpaf(args):
    args.figure_width = 10
    args.dpi_factor = 1.0
    args.keypoint_threshold_rel = 0.0
    args.instance_threshold = 0.2
    args.keypoint_threshold = 0

    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    processor, net = processor_factory(args)
    preprocess = preprocess_factory(args)
    return net, processor, preprocess

"""
def load_pifpaf(device, version='shufflenetv2k16w'):
    print('Loading Pifpaf')
    net_cpu, _ = openpifpaf.network.Factory(checkpoint=version, download_progress=False).factory()
    net = net_cpu.to(device)
    openpifpaf.decoder.CifSeeds.threshold = 0.5
    openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.0
    openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
    openpifpaf.decoder.nms.Keypoints.keypoint_threshold_rel = 0.0
    openpifpaf.decoder.CifCaf.force_complete = True
    processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)
    preprocess = openpifpaf.transforms.Compose([openpifpaf.transforms.NormalizeAnnotations(),openpifpaf.transforms.CenterPadTight(16),openpifpaf.transforms.EVAL_TRANSFORM])
    return net, processor, preprocess"""