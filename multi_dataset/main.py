from dataset import *
from network import *
from utils import *
import argparse
import matplotlib.pyplot as plt
import copy
import pickle

# Parameters

parser = argparse.ArgumentParser(description='Cross dataset training and testing')

INPUT_SIZE = 51 # 3*17 (x + y + confidence)
OUTPUT_SIZE = 1
LINEAR_SIZE=256
P_DROPOUT=0.2
NUM_STAGES=3
LEARNING_RATE=0.0001
EPOCHS = 50
BATCH_SIZE=512
PATH_MODEL = '/home/caristan/code/looking/annotator/models/'
training = True
normalize = True
# Cuda

np.random.seed(1)
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)

parser.add_argument('-train_pr', '--tr_pr', type=str, help='proportion for the training', default="kitti,jack,jaad,pie")
parser.add_argument('-test_pr', '--ts_pr', type=str, help='proportion for the training', default="kitti,jack,jaad,pie")

args = parser.parse_args()

in_datas = []
out_datas = []

in_datas_test = []
out_datas_test = []

train_ = ''
test_ = ''

not_random = False

if args.tr_pr:
	text_s = parse_str(args.tr_pr)
	if 'kitti' in text_s:
		KittiData = Dataset('/home/caristan/code/looking/looking/data/train_kitti.json', True)

		joints_kitti = KittiData.get_joints()
		labels_kitti = KittiData.get_outputs().view(-1,1).type(torch.float)


		in_datas.append(joints_kitti)
		out_datas.append(labels_kitti)
		train_ += 'kitti '

	if 'jack' in text_s:
		JackData = Dataset('/home/caristan/code/looking/looking/data/train_jack.json', True)

		joints_JACK = JackData.get_joints()
		labels_JACK = JackData.get_outputs().view(-1,1).type(torch.float)

		in_datas.append(joints_JACK)
		out_datas.append(labels_JACK)
		train_ += 'jack '

	if 'nu' in text_s:
		NuData = Dataset('/home/caristan/code/looking/looking/data/train_nu.json', True)

		joints_NU = NuData.get_joints()
		labels_NU = NuData.get_outputs().view(-1,1).type(torch.float)

		in_datas.append(joints_NU)
		out_datas.append(labels_NU)
		train_ += 'nu '

	if 'jaad' in text_s:
		if not_random:
			LookingData_ = LookingDataset_balanced('/home/caristan/code/looking/looking/data/jaad_all.json', 'train', False, normalize, False, False)
			#LookingData_ = LookingDataset_balanced_random('/home/younesbelkada/Travail/test_looking/data/data_scale1_force.json', 'train', False, normalize, False, False)
			joints_train_JAAD, outputs_train_JAAD, _ = LookingData_.get_train()
			in_datas.append(joints_train_JAAD)
			out_datas.append(outputs_train_JAAD.view(-1,1))

			train_ += 'jaad '
			if 'jaad' in args.ts_pr.split(','):
				joints_test_JAAD, outputs_test_JAAD, _ = LookingData_.get_test()
				in_datas_test.append(joints_test_JAAD)
				out_datas_test.append(outputs_test_JAAD.view(-1,1))
				test_ += 'jaad'

		else:
			LookingData_ = LookingDataset_balanced_random('/home/caristan/code/looking/looking/data/jaad_all.json', 'train', False, normalize, False, False)
			joints_train_JAAD, outputs_train_JAAD, _ = LookingData_.get_train()
			in_datas.append(joints_train_JAAD)
			out_datas.append(outputs_train_JAAD.view(-1,1))
			train_ += 'jaad '
			if 'jaad' in args.ts_pr.split(','):
				joints_test_JAAD, outputs_test_JAAD, _ = LookingData_.get_test()
				in_datas_test.append(joints_test_JAAD)
				out_datas_test.append(outputs_test_JAAD.view(-1,1))
				test_ += 'jaad'

	if 'pie' in text_s:
		data = PIE_Dataset("/home/caristan/code/looking/looking/data/pie_all.json")

		joints_train_PIE, outputs_train_PIE, _ = data.get_train()
		

		in_datas.append(joints_train_PIE)
		out_datas.append(outputs_train_PIE.view(-1,1))
		train_ += 'pie '
		if 'pie' in args.ts_pr.split(','):
			joints_test_PIE, outputs_test_PIE, _ = data.get_test()
			in_datas_test.append(joints_test_PIE)
			out_datas_test.append(outputs_test_PIE.view(-1,1))
			test_ += "pie"

if args.ts_pr:
	text_s = parse_str(args.ts_pr)
	if 'kitti' in text_s:
		KittiData_test = Dataset('/home/caristan/code/looking/looking/data/test_kitti.json', True)

		joints_kitti_test = KittiData_test.get_joints()
		labels_kitti_test = KittiData_test.get_outputs().view(-1,1).type(torch.float)

		in_datas_test.append(joints_kitti_test)
		out_datas_test.append(labels_kitti_test)
		test_ += "kitti "

	if 'jack' in text_s:
		JackData_test = Dataset('/home/caristan/code/looking/looking/data/test_jack.json', True)

		joints_test_JACK = JackData_test.get_joints()
		labels_test_JACK = JackData_test.get_outputs().view(-1,1).type(torch.float)

		in_datas_test.append(joints_test_JACK)
		out_datas_test.append(labels_test_JACK)
		test_ += "jack "

	if 'nu' in text_s:
		NuData_test = Dataset('/home/caristan/code/looking/looking/data/test_nu.json', True)

		joints_test_NU = NuData_test.get_joints()
		labels_test_NU = NuData_test.get_outputs().view(-1,1).type(torch.float)

		in_datas_test.append(joints_test_NU)
		out_datas_test.append(labels_test_NU)
		test_ += "nu "

	if 'jaad' in text_s and 'jaad' not in args.tr_pr.split(','):
		LookingData_ = LookingDataset_balanced('/home/caristan/code/looking/looking/data/jaad_all.json', 'train', False, normalize, False, False)

		joints_train_JAAD, outputs_train_JAAD, _ = LookingData_.get_train()
		joints_test_JAAD, outputs_test_JAAD, _ = LookingData_.get_test()

		in_datas.append(joints_train_JAAD)
		out_datas.append(outputs_train_JAAD.view(-1,1))

		in_datas_test.append(joints_test_JAAD)
		out_datas_test.append(outputs_test_JAAD.view(-1,1))
		test_ += "jaad "

	if 'pie' in text_s and 'pie' not in args.tr_pr.split(','):
		data = PIE_Dataset("/home/caristan/code/looking/looking/data/pie_all.json")

		joints_test_PIE, outputs_test_PIE, _ = data.get_test()

		in_datas_test.append(joints_test_PIE)
		out_datas_test.append(outputs_test_PIE.view(-1,1))
		test_ += "pie "


joints_train = torch.cat(tuple(in_datas), dim=0)
labels_train = torch.cat(tuple(out_datas), dim=0)

joints_test = torch.cat(tuple(in_datas_test), dim=0)
labels_test = torch.cat(tuple(out_datas_test), dim=0)

FinalData = Data(joints_train, labels_train)
FinalData_test = Data(joints_test, labels_test)


print("Training proportion : {} ".format(train_))

print("Testing proportion : {} ".format(test_))


"""

JackData = Dataset('./data/train_jack.json', True)
JackData_test = Dataset('./data/test_jack.json', True)



data = PIE_Dataset("/home/younesbelkada/Travail/PIE/final_data_new.json")
LookingData_ = LookingDataset_balanced('/home/younesbelkada/Travail/test_looking/data/data_scale1_force.json', 'train', False, normalize, False, False)

joints_train_PIE, outputs_train_PIE, names_train_PIE = data.get_train()
joints_test_PIE, outputs_test_PIE, names_test_PIE = data.get_test()

joints_train_JAAD, outputs_train_JAAD, names_train_JAAD = LookingData_.get_train()
joints_test_JAAD, outputs_test_JAAD, names_test_JAAD = LookingData_.get_test()

joints_JACK = JackData.get_joints()
labels_JACK = JackData.get_outputs().view(-1,1).type(torch.float)

joints_test_JACK = JackData_test.get_joints()
labels_test_JACK = JackData_test.get_outputs().view(-1,1).type(torch.float)


#exit(0)
PATH_MODEL = './models/'


joints_train = torch.cat((joints_train_PIE, joints_train_JAAD, joints_JACK), dim=0)
labels_train = torch.cat((outputs_train_PIE.view(-1,1), outputs_train_JAAD.view(-1,1), labels_JACK), dim=0)

joints_test = torch.cat((joints_test_PIE, joints_test_JAAD, joints_test_JACK), dim=0)
labels_test = torch.cat((outputs_test_PIE.view(-1,1), outputs_test_JAAD, labels_test_JACK), dim=0)

FinalData = Data(joints_train, labels_train)
FinalData_test = Data(joints_test, labels_test)
"""

train_loader = DataLoader(dataset=FinalData, batch_size=BATCH_SIZE, shuffle=True)

joints = FinalData.get_joints()
joints_test = FinalData_test.get_joints()

labels = FinalData.get_outputs()
labels_test = FinalData_test.get_outputs()


print("There are :{} instances and {} positive samples on the test set ".format(labels_test.shape[0], sum(labels_test).item()))


# Train
if training:
	loss = nn.BCELoss() # maybe remove the sigmoid at the last layer....
	train_acc = []
	train_aps = []
	test_ap = []
	test_ac = []
	model = LookingModel(INPUT_SIZE, OUTPUT_SIZE, LINEAR_SIZE, P_DROPOUT, NUM_STAGES, False).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	model.train()
	m = nn.Softmax(dim=0)
	grads = []
	grads_x = []
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

		acc = binary_acc(out_batch, labels)
		average_p = average_precision(out_batch, labels)

		acc_test = binary_acc(out_test, labels_test)
		ap_test = average_precision(out_test, labels_test)

		acc0, acc1 = acc_per_class(out_batch, labels)
		train_aps.append(average_p)
		train_acc.append(acc)
		test_ap.append(ap_test)
		test_ac.append(acc_test)
		#print(acc0, acc1)
		print_summary(epoch+1, EPOCHS, loss(model(joints), labels).item(), acc, acc_test, average_p, ap_test, acc1, acc0)
		model.train()
	model.eval()
	torch.save(model, PATH_MODEL+'/looking_model_jack.pkl')
	print('')
	print('Model saved to disk')
	print("Best AP : {}".format(max(test_ap)))
	i = test_ap.index(max(test_ap))
	print("Best Ac : {}".format(test_ac[i]))
