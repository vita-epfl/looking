from dataset_example import *
import torchvision.transforms.functional as F

class SquarePad:
	"""
	Class to apply the sqaure pad resizing method
	"""
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

data_transform = transforms.Compose([
		SquarePad(),
        transforms.ToTensor(),
	transforms.ToPILImage(),
        transforms.Resize((224,224)),
	transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

data = Example_Dataset('train', './', 'jack', data_transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
for head, joints, labels in data_loader:
	print(head.shape)
	print(joints.shape)
	print(labels.shape)
	break

data = Example_Dataset('train', './', 'nu', data_transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
for head, joints, labels in data_loader:
	print(head.shape)
	print(joints.shape)
	print(labels.shape)
	break