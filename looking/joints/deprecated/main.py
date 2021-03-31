from network import *
from dataset import *
import torch
import torch.nn as nn
import argparse

torch.manual_seed(0)


# Parser 

parser = argparse.ArgumentParser(description='Training and evaluating on Kitti')

# parameters

parser.add_argument('-epochs', '--e', type=int, help='number of epochs for training', default=30)
parser.add_argument('-learning_rate', '--lr', type=float, help='learning rate for training', default=0.0001)
parser.add_argument('-split', '--split', type=str, help='dataset split', default="original")
parser.add_argument('-kt', help='evaluate on kitti', action='store_true')
parser.add_argument('-droupout', '--dr', type=float, help='dropout rate for training', default=0.2)
parser.add_argument('-path', '--pt', type=str, help='path for model saving', default='./models/')
parser.add_argument('-pose', '--p', type=str, help='pose type', default="full")

args = parser.parse_args()

EPOCHS = args.e
video = args.split
pose = args.p

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)

if pose == "head":
	INPUT_SIZE = 15
elif pose == "body":
	INPUT_SIZE = 36
else:
	INPUT_SIZE = 51


jaad_train = JAAD_Dataset_joints("train", pose, video)
jaad_val = JAAD_Dataset_joints("val", pose, video)


P_DROPOUT=args.dr
LEARNING_RATE= args.lr
PATH_MODEL = args.pt



training = True
inf = args.kt


joints, labels = jaad_train.get_joints()
joints_test, labels_test = jaad_val.get_joints()

model = LookingModel(INPUT_SIZE)
train_loader = torch.utils.data.DataLoader(jaad_train, batch_size=128, shuffle=True)

best_ap = 0

if training:
	loss = nn.BCELoss() # maybe remove the sigmoid at the last layer....
	train_acc = []
	train_aps = []
	test_ap = []
	test_ac = []
	#model = LookingModel(INPUT_SIZE, OUTPUT_SIZE, LINEAR_SIZE, P_DROPOUT, NUM_STAGES, False).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	model.train()
	for epoch in range(EPOCHS):
		model.to(device)
		for x_batch, y_batch in train_loader:
			if torch.cuda.is_available():
				#print('heho')
				x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
				#x_batch.requires_grad=True
			optimizer.zero_grad()
			output = model(x_batch.float())
			error = loss(output, y_batch.view(-1,1))
			error.backward()
			optimizer.step()
		model.eval()
		torch.cuda.empty_cache()
		model.to("cpu")
		optimizer.zero_grad()

		out_batch = model(joints)
		out_test = model(joints_test)

		acc = binary_acc(out_batch, labels.view(-1,1))
		average_p = average_precision(out_batch, labels)

		acc_test = binary_acc(out_test, labels_test.view(-1,1))
		ap_test = average_precision(out_test, labels_test)

		acc0, acc1 = acc_per_class(out_batch, labels)
		train_aps.append(average_p)
		train_acc.append(acc)
		test_ap.append(ap_test)
		test_ac.append(acc_test)
		#print(acc0, acc1)


		print_summary(epoch+1, EPOCHS, loss(model(joints), labels.float().view(-1,1)).item(), acc, acc_test, average_p, ap_test, acc1, acc0)
		model.train()
		if ap_test > best_ap:
			best_ap = ap_test
			best_ac = acc_test
			torch.save(model, './models/looking_model_jaad_{}_{}_kps.p'.format(video, pose))



print()
model = []
model = torch.load("./models/looking_model_jaad_{}_{}_kps.p".format(video, pose), map_location=torch.device(device))
model.eval()	
jaad_test = JAAD_Dataset_joints("test", pose, video)
joints_test, labels_test = jaad_test.get_joints()
out_test = model(joints_test.to(device))
acc_test = binary_acc(out_test.to(device), labels_test.view(-1,1).to(device))
ap_test = average_precision(out_test.to(device), labels_test.to(device))
print("Best AP on JAAD test set : {} | {}".format(ap_test, acc_test))

if inf:
	model = []
	model = torch.load("./models/looking_model_jaad_{}_{}_kps.p".format(video, pose), map_location=torch.device(device))
	jaad_val = Kitti_Dataset_joints("test", pose)
	model.eval()

	joints_test, labels_test = jaad_val.get_joints()

	out_test = model(joints_test.to(device))
	acc_test = binary_acc(out_test.to(device), labels_test.view(-1,1).to(device))
	ap_test = average_precision(out_test.to(device), labels_test.to(device))

	print("Kitti | AP : {} | Acc : {}".format(ap_test, acc_test))


