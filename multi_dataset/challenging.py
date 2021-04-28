import openpifpaf
import torch
import PIL
import numpy as np
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("device : ", device)

def load_pifpaf():
    print('Loading Pifpaf')
    net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k16w', download_progress=False)
    net = net_cpu.to(device)
    openpifpaf.decoder.CifSeeds.threshold = 0.5
    openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.0
    openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
    openpifpaf.decoder.nms.Keypoints.keypoint_threshold_rel = 0.0
    openpifpaf.decoder.CifCaf.force_complete = True
    processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)
    preprocess = openpifpaf.transforms.Compose([openpifpaf.transforms.NormalizeAnnotations(),openpifpaf.transforms.CenterPadTight(16),openpifpaf.transforms.EVAL_TRANSFORM])
    return net, processor, preprocess


net, processor, preprocess = load_pifpaf()

file = open('to_annotate_back.txt', 'w')

path = "/work/vita/datasets/NUSCENES/US/sweeps/"

tab = os.listdir('./')

def count(path):
        pil_im = PIL.Image.open(path).convert('RGB')
        im = np.asarray(pil_im)
        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)
        loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True, collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        i = 0
	for images_batch, _, __ in loader:
                predictions = processor.batch(net, images_batch, device=device)[0]
                break
        #print(len(predictions))
        if len(predictions) > 6:
                file.write(path+'\n')
                return 1
        return 0
i = 0

for t in tab:
	if t[-1] != 'y' and t[-1] != 'n' and t[-1] != 't':
                tab2 = os.listdir(path+t+'/')
                for im in tab2:
                        i += count(path+t+'/'+im)
                        if i >= 2000:
                                break
file.close()

