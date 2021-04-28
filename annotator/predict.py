import PyQt5
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QLabel
from glob import glob
import os
from utils_predict import *
import openpifpaf
import PIL


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)



def load_pifpaf():
	print('Loading Pifpaf')
	net_cpu, _ = openpifpaf.network.factory.Factory(checkpoint='shufflenetv2k30', download_progress=False).factory()
	net = net_cpu.to(device)
	openpifpaf.decoder.utils.CifSeeds.threshold = 0.5
	openpifpaf.decoder.utils.nms.Keypoints.keypoint_threshold = 0.0
	openpifpaf.decoder.utils.nms.Keypoints.instance_threshold = 0.1
	openpifpaf.decoder.utils.nms.Keypoints.keypoint_threshold_rel = 0.0
	openpifpaf.decoder.CifCaf.force_complete = True
	decoder = openpifpaf.decoder.factory([hn.meta for hn in net_cpu.head_nets])
	preprocess = openpifpaf.transforms.Compose([
	openpifpaf.transforms.NormalizeAnnotations(),
	openpifpaf.transforms.RescaleAbsolute(long_edge=2500),
	openpifpaf.transforms.CenterPadTight(16),
	openpifpaf.transforms.EVAL_TRANSFORM,
	])
	return net, decoder, preprocess

class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(240, 320)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
		self.verticalLayoutWidget.setGeometry(QtCore.QRect(40, 90, 160, 80))
		self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
		self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
		self.verticalLayout.setContentsMargins(0, 0, 0, 0)
		self.verticalLayout.setObjectName("verticalLayout")
		self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
		self.pushButton_2.setObjectName("pushButton_2")
		self.verticalLayout.addWidget(self.pushButton_2)
		self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
		self.pushButton.setObjectName("pushButton")
		self.verticalLayout.addWidget(self.pushButton)
		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 240, 20))
		self.menubar.setObjectName("menubar")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)


		self.selected = False
		self.path = ""
		self.predictor = Predictor(self.path)

		self.pushButton_2.clicked.connect(self.predict)
		self.pushButton.clicked.connect(self.select_folder)

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
		self.pushButton_2.setText(_translate("MainWindow", "Predict"))
		self.pushButton.setText(_translate("MainWindow", "Select Folder"))

	def predict(self):
		if not self.selected:
			alert = QtWidgets.QMessageBox()
			alert.setText('Please Load a non empty folder before')
			alert.exec_()
		else:
			reply = QtWidgets.QMessageBox.question(QtWidgets.QMainWindow(), 'Continue?', 'Are you sure ? All your saved annotations will be erased.', QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
			if reply == QtWidgets.QMessageBox.Yes:
				self.predictor.create_folders()
				self.predictor.predict()
				alert = QtWidgets.QMessageBox()
				alert.setText('Predictions completed, please close this program !')
				alert.exec_()
				self.selected = False

	def select_folder(self):
		self.path = str(QFileDialog.getExistingDirectory(self.pushButton, "Select Directory"))
		if len(glob(self.path+'/*.png')+glob(self.path+'/*.jpg')) > 0:
			self.selected = True
			self.predictor.path = self.path
		else:
			print("WARNING: You have selected an empty folder")


class Predictor():
	def __init__(self, path):
		self.path = path

		self.model = torch.load("./models/looking_model_jaad_video_full_kps_romain.p", map_location=torch.device(device))
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
		for image in sorted(glob(self.path+'/*.png')+glob(self.path+'/*.jpg')):
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
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setWindowIcon(QtGui.QIcon('logo.png'))
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

#Predictor(path)
