from glob import glob, iglob
import os
import openpifpaf
import PIL
import torch
import numpy as np
import json
import cv2
import argparse



parser = argparse.ArgumentParser(description='Running pifpaf in folder')
parser.add_argument('--path', '-p', type=str, help='the folder path')
parser.add_argument('--output', '-o', type=str, help='the output folder path')

args = parser.parse_args()

def rectangle(img, data):
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 0.30
	fontColor              = (0,0,0)
	lineType               = 1

	blk = np.zeros(img.shape, np.uint8)
	look = []
	bboxes = []
	for i in range(len(data)):
		bb = data[i]['bbox']
		bboxes.append(bb)

		blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (0,0,0), 1)
		blk = cv2.rectangle(blk, (int(bb[0]), int(bb[1])-10), (int(bb[0]+30), int(bb[1])), (0,0,0), -1)

	img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
	return img, bboxes



folder_to_run = args.path
if not args.output:
	output_folder = folder_to_run + '/output_pifpaf_jaad'
else:
	output_folder = args.output


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)


print('Loading Pifpaf')
net_cpu, _ = openpifpaf.network.factory.Factory(checkpoint='shufflenetv2k30', download_progress=False).factory()
net = net_cpu.to(device)
openpifpaf.decoder.utils.CifSeeds.threshold = 0.5
openpifpaf.decoder.utils.nms.Keypoints.keypoint_threshold = 0.0
openpifpaf.decoder.utils.nms.Keypoints.instance_threshold = 0.2
openpifpaf.decoder.utils.nms.Keypoints.keypoint_threshold_rel = 0.0
openpifpaf.decoder.CifCaf.force_complete = True
processor = openpifpaf.decoder.factory([hn.meta for hn in net_cpu.head_nets])
preprocess = openpifpaf.transforms.Compose([
openpifpaf.transforms.NormalizeAnnotations(),
openpifpaf.transforms.CenterPadTight(16),
openpifpaf.transforms.EVAL_TRANSFORM,
])

for folder, _, _ in os.walk(args.path):
	out_fol = output_folder + '/' + folder.split('/')[-1]
	if not os.path.exists(out_fol):
		os.makedirs(out_fol)
		for image in sorted(glob(folder + '/*.png', recursive=True) + glob(folder + '/*.jpg', recursive=True)):
			pil_im = PIL.Image.open(image).convert('RGB')
			im = np.asarray(pil_im)
			data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)
			loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True, collate_fn=openpifpaf.datasets.collate_images_anns_meta)
			for images_batch, _, __ in loader:
		    		predictions = processor.batch(net_cpu, images_batch, device=device)[0]
		    		tab_predict = [p.json_data() for p in predictions]
			with open(out_fol + '/{}.predictions.json'.format(image.split('/')[-1]), 'w') as outfile:
		    		json.dump(tab_predict, outfile)
			
