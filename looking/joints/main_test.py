
from network import *
from dataset import *
import torch
import torch.nn as nn
import argparse
from losses import FocalLoss
torch.manual_seed(0)


# Parser 

parser = argparse.ArgumentParser(description='Training and evaluating on Kitti')

# parameters

parser.add_argument('--epochs', '-e', type=int, help='number of epochs for training', default=30)
parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate for training', default=0.0001)
parser.add_argument('--split', type=str, help='dataset split', default="video")
parser.add_argument('--kitti', help='evaluate on kitti', action='store_true')
parser.add_argument('--dropout', type=float, help='dropout rate for training', default=0.2)
parser.add_argument('--out_path', type=str, help='path for model saving', default='./models/')
parser.add_argument('--pose',  type=str, help='pose type', default="full")
parser.add_argument('--batch_size', '-bs', help='enable focal loss', default=128)
parser.add_argument('--focal_loss', help='enable focal loss', action='store_true')

args = parser.parse_args()

EPOCHS = args.epochs
video = args.split
pose = args.pose

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)
if pose == "head":
	INPUT_SIZE = 15
elif pose == "body":
	INPUT_SIZE = 36
else:
	INPUT_SIZE = 51


jaad_train = JAAD_Dataset_joints_new("../data/", "JAAD_2k30/", "train", pose, video)
jaad_val = JAAD_Dataset_joints_new("../data/", "JAAD_2k30/", "val", pose, video)



P_DROPOUT=args.dropout
LEARNING_RATE= args.learning_rate
PATH_MODEL = args.path


training = True
kitti = args.kitti

joints, labels = jaad_train.get_joints()
joints_test, labels_test = jaad_val.get_joints()

model = LookingModel(INPUT_SIZE).to(device)
train_loader = torch.utils.data.DataLoader(jaad_train, batch_size=128, shuffle=True)
best_ap, best_ac, best_epoch = 0, 0, 0

if training:
	# loss = nn.BCELoss()
	if args.focal_loss:
		criterion = FocalLoss(alpha=1, gamma=3)
	else:
		criterion = nn.BCELoss()
	train_acc = []
	train_aps = []
	test_ap = []
	test_ac = []
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	model.train()
	for epoch in range(EPOCHS):
		model.to(device)
		running_loss = 0
		for x_batch, y_batch in train_loader:
			if torch.cuda.is_available():
				x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
				#x_batch.requires_grad=True
			optimizer.zero_grad()
			output = model(x_batch.float())
			loss = criterion(output, y_batch.view(-1, 1))
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

		#acc_test = binary_acc(out_test, labels_test.view(-1,1))
		#ap_test = average_precision(out_test, labels_test)

		ap_test, acc_test = jaad_val.evaluate(model, device)

		acc0, acc1 = acc_per_class(out_batch, labels)
		train_aps.append(average_p)
		train_acc.append(acc)
		test_ap.append(ap_test)
		test_ac.append(acc_test)
		#print(acc0, acc1)
		running_loss /= len(jaad_train)

		print_summary(epoch+1, EPOCHS,  running_loss, acc, acc_test, average_p, ap_test, acc1, acc0)
		# print_summary(epoch + 1, EPOCHS, criterion(model(joints.to(device)).to(device), labels.float().view(-1, 1).to(device)).item(),
		# 			  acc, acc_test, average_p, ap_test, acc1, acc0)

		model.train()
		if ap_test > best_ap:
			best_epoch = epoch
			best_ap = ap_test
			best_ac = acc_test
			torch.save(model, './models/looking_model_jaad_{}_{}_kps.p'.format(video, pose))


print()
print(f'Best epoch selected from validation: {best_epoch} with an accuracy of {best_ap*100:.1f}% ')
print("Starting evaluation ...")
model = []
model = torch.load("./models/looking_model_jaad_{}_{}_kps.p".format(video, pose), map_location=torch.device(device))
model.eval()	
jaad_test = JAAD_Dataset_joints_new("../data/", "JAAD_2k30/", "test", pose, video)

ap_test, acc_test = jaad_test.evaluate(model, device, 10)

#joints_test, labels_test = jaad_test.get_joints()
#out_test = model(joints_test.to(device))
#acc_test = binary_acc(out_test.to(device), labels_test.view(-1,1).to(device))
#ap_test = average_precision(out_test.to(device), labels_test.to(device))
print("Best AP on JAAD test set : {:.1f}%".format(ap_test*100))

if kitti:
	model = []
	model = torch.load("./models/looking_model_jaad_{}_{}_kps.p".format(video, pose), map_location=torch.device(device))
	jaad_val = Kitti_Dataset_joints("test", pose)
	model.eval()

	joints_test, labels_test = jaad_val.get_joints()

	out_test = model(joints_test.to(device))
	acc_test = binary_acc(out_test.to(device), labels_test.view(-1,1).to(device))
	ap_test = average_precision(out_test.to(device), labels_test.to(device))

	print("Kitti | AP : {} | Acc : {}".format(ap_test, acc_test))


