from network import *
from dataset import *
import torch
import torch.nn as nn
import argparse

torch.manual_seed(0)


# Parser

parser = argparse.ArgumentParser(description='Cross dataset jaad')

# parameters

parser.add_argument('--train', '-t', type=str, help='proportion for the training', default="jaad,kitti,jack,jaad,pie")
parser.add_argument('--split_path', '-jsp', type=str, help='proportion for the training', default="/home/caristan/code/looking/looking/splits/")
parser.add_argument('--data_path', '-dp', type=str, help='proportion for the training', default="../data/")


args = parser.parse_args()

split_path = args.jaad_split_path
data_path = args.data_path

EPOCHS = 50

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)

INPUT_SIZE = 51


jaad_train = JAAD_Dataset_joints(data_path, split_path, "train")
jaad_val = JAAD_Dataset_joints(data_path, split_path, "val")


LEARNING_RATE = 0.0001


training = True


in_datas = []
out_datas = []

in_datas_test = []
out_datas_test = []

train_ = ''
test_ = ''

joints, labels = jaad_train.get_joints()
joints_test, labels_test = jaad_val.get_joints()

in_datas.append(joints)
out_datas.append(labels.view(-1,1))

in_datas_test.append(joints_test)
out_datas_test.append(labels_test.view(-1,1))

if args.train:
	text_s = parse_str(args.tr_pr)
	if 'kitti' in text_s:
		KittiData = Dataset('./data/train_kitti.json', True)

		joints_kitti = KittiData.get_joints()
		labels_kitti = KittiData.get_outputs().view(-1,1).type(torch.float)


		in_datas.append(joints_kitti)
		out_datas.append(labels_kitti)
		train_ += 'kitti '

	if 'jack' in text_s:
		JackData = Dataset('./data/train_jack.json', True)

		joints_JACK = JackData.get_joints()
		labels_JACK = JackData.get_outputs().view(-1,1).type(torch.float)

		in_datas.append(joints_JACK)
		out_datas.append(labels_JACK)
		train_ += 'jack '

	if 'nu' in text_s:
		NuData = Dataset('./data/Nuscenes_train.json', True)

		joints_NU = NuData.get_joints()
		labels_NU = NuData.get_outputs().view(-1,1).type(torch.float)

		in_datas.append(joints_NU)
		out_datas.append(labels_NU)
		train_ += 'nu '


	if 'pie' in text_s:
		data = PIE_Dataset("./data/pie_all.json")

		joints_train_PIE, outputs_train_PIE, _ = data.get_train()


		in_datas.append(joints_train_PIE)
		out_datas.append(outputs_train_PIE.view(-1,1))
		train_ += 'pie '


joints_train = torch.cat(tuple(in_datas), dim=0)
labels_train = torch.cat(tuple(out_datas), dim=0)

joints_test = torch.cat(tuple(in_datas_test), dim=0)
labels_test = torch.cat(tuple(out_datas_test), dim=0)

FinalData = Data(joints_train, labels_train)
FinalData_test = Data(joints_test, labels_test)


model = LookingModel(INPUT_SIZE).to(device)
train_loader = torch.utils.data.DataLoader(FinalData, batch_size=128, shuffle=True)
best_ap, best_ac, best_epoch = 0, 0, 0

if training:
	criterion = nn.BCELoss()
	train_acc = []
	train_aps = []
	test_ap = []
	test_ac = []
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	model.train()
	for epoch in range(EPOCHS):
		i = 0
		model.to(device)
		running_loss = 0
		for x_batch, y_batch in train_loader:
			if torch.cuda.is_available():
				x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
			optimizer.zero_grad()
			output = model(x_batch.float())
			print(output.type())
			loss = criterion(output, y_batch.view(-1, 1).to(torch.float))
			running_loss += loss.item() * x_batch.size(0)
			loss.backward()
			optimizer.step()
		model.eval()
		torch.cuda.empty_cache()
		model.to("cpu")

		out_batch = model(joints)
		out_test = model(joints_test)

		acc = binary_acc(out_batch, labels.view(-1,1))
		average_p = average_precision(out_batch, labels)
		ap_test, acc_test = jaad_val.evaluate(model, device)

		acc0, acc1 = acc_per_class(out_batch, labels)
		train_aps.append(average_p)
		train_acc.append(acc)
		test_ap.append(ap_test)
		test_ac.append(acc_test)
		running_loss /= len(FinalData)

		print_summary(epoch+1, EPOCHS,  running_loss, acc, acc_test, average_p, ap_test, acc1, acc0)

		model.train()
		if ap_test > best_ap:
			best_epoch = epoch
			best_ap = ap_test
			best_ac = acc_test



print()
print(f'Best epoch selected from validation: {best_epoch} with an accuracy of {best_ac*100:.1f}% and AP of {best_ap*100:.1f}')
print("Starting evaluation ...")
model.eval()
jaad_test = JAAD_Dataset_joints_new(data_path, split_path, "test")

ap_test, acc_test = jaad_test.evaluate(model, device, 10)
print("Best AP on JAAD test set : {:.1f}%".format(ap_test*100))
