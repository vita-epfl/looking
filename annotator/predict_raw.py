import os
from utils_predict import *
import openpifpaf
import PIL
from glob import glob
import torch
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)

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

class Predictor():
	def __init__(self, path):
		self.path = path

		self.model = torch.load("./models/looking_model_jack.pkl", map_location=torch.device(device))
		self.model.eval()

		self.net, self.processor, self.preprocess = load_pifpaf()

		#self.predict()

	def create_folders(self):
		self.directory_look = self.path+'/look_'+self.path.split('/')[-1]
		if not os.path.exists(self.directory_look):
			os.makedirs(self.directory_look)
		self.directory_out = self.path+'/out_'+self.path.split('/')[-1]
		if not os.path.exists(self.directory_out):
			os.makedirs(self.directory_out)
		self.directory_anno = self.path+'/anno_'+self.path.split('/')[-1]
		if not os.path.exists(self.directory_anno):
			os.makedirs(self.directory_anno)

	def predict(self):
		for image in tqdm(sorted(glob(self.path+'/*.png')+glob(self.path+'/*.jpg'))):
			pil_im = PIL.Image.open(image).convert('RGB')
			im = np.asarray(pil_im)
			data = openpifpaf.datasets.PilImageList([pil_im], preprocess=self.preprocess)
			loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True, collate_fn=openpifpaf.datasets.collate_images_anns_meta)
			for images_batch, _, __ in loader:
				predictions = self.processor.batch(self.net, images_batch, device=device)[0]
				tab_predict = [p.json_data() for p in predictions]
			with open(self.directory_out+'/{}.predictions.json'.format(image.split('/')[-1]), 'w') as outfile:
				json.dump(tab_predict, outfile)
			with open(self.directory_out+'/{}.predictions.json'.format(image.split('/')[-1]), 'r') as file:
				data = json.load(file)
			img = cv2.imread(image)
			if os.path.exists(self.directory_anno):
				if image.split('/')[-1]+'.json' not in os.listdir(self.directory_anno):
					img_out, Y, X, bboxes = run_and_rectangle(img, data, self.model, device)
				else:
					data2 = json.load(open(self.directory_anno+'/{}.json'.format(image.split('/')[-1]), 'r'))
					if len(data) == len(data2["Y"]):
						img_out, Y, X, bboxes = run_and_rectangle_saved(img, data, self.model, device, data2)
					else:
						img_out, Y, X, bboxes = run_and_rectangle(img, data, self.model, device)
			else:
				img_out, Y, X, bboxes = run_and_rectangle(img, data, self.model, device)

			cv2.imwrite(self.directory_look+'/'+image.split('/')[-1], img_out)
			data = {'X':X, 'Y':Y, 'bbox':bboxes}
			with open(self.directory_anno+'/'+image.split('/')[-1]+'.json', 'w') as file:
				json.dump(data, file)


if __name__ == "__main__":
	import sys
	if len(sys.argv) != 2:
	    print("Usage predict_raw.py path") 
	else:
	    path = sys.argv[1]
	    print("Launching the predictor on the folder : {}".format(path))
	    predictor = Predictor(path)
	    predictor.create_folders()
	    predictor.predict()