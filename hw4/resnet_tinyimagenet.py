import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim     
from torch.autograd import Variable 

class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(BasicBlock, self).__init__()
		self.residual = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels))

		self.shortcut = nn.Sequential()
		if stride!=1 or in_channels!=out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels))

	def forward(self, x):
		return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))

class ResNet(nn.Module):
	def __init__(self, basic_block, num_blocks, num_classes):
		super(ResNet, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5))
		self.conv2_x = self._make_layer(basic_block, 32, 32, num_blocks[0], 1)
		self.conv3_x = self._make_layer(basic_block, 32, 64, num_blocks[1], 2)
		self.conv4_x = self._make_layer(basic_block, 64, 128, num_blocks[2], 2)
		self.conv5_x = self._make_layer(basic_block, 128, 256, num_blocks[3], 2)
		self.pool = nn.MaxPool2d(4, 4)
		self.fc = nn.Linear(256, num_classes)

	def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(in_channels, out_channels, stride))
			in_channels = out_channels

		return nn.Sequential(*layers)

	def forward(self, x):
		output = self.conv1(x)
		output = self.conv2_x(output)
		output = self.conv3_x(output)
		output = self.conv4_x(output)
		output = self.conv5_x(output)
		output = self.pool(output)
		output = output.view(output.size(0), -1)
		output = self.fc(output)

		return output 

def create_val_folder(val_dir):
	"""
	This method is responsible for separating validation
	images into separate sub folders
	"""
	# path where validation data is present now
	path = os.path.join(val_dir, 'images')
	# file where image2class mapping is present
	filename = os.path.join(val_dir, 'val_annotations.txt')
	fp = open(filename, "r") # open file in read mode
	data = fp.readlines() # read line by line
	'''
	Create a dictionary with image names as key and
	corresponding classes as values
	'''
	val_img_dict = {}
	for line in data:
		words = line.split("\t")
		val_img_dict[words[0]] = words[1]
	fp.close()
	# Create folder if not present, and move image into proper folder
	for img, folder in val_img_dict.items():
		newpath = (os.path.join(path, folder))
		if not os.path.exists(newpath): # check if folder exists
			os.makedirs(newpath)
		# Check if image exists in default directory
		if os.path.exists(os.path.join(path, img)):
			os.rename(os.path.join(path, img), os.path.join(newpath, img))
	return

TINY_TRAIN_MEAN, TINY_TRAIN_STD =(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
	transforms.RandomCrop(64, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(15),
	transforms.ToTensor(),
	transforms.Normalize(TINY_TRAIN_MEAN, TINY_TRAIN_STD)
	])

train_dir = '/u/training/tra169/scratch/tiny-imagenet-200/train'
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dir = '/u/training/tra169/scratch/tiny-imagenet-200/val/images'
if 'val_' in os.listdir(val_dir)[0]:
	create_val_folder(val_dir)
else:
	pass
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

net = ResNet(BasicBlock,[2,4,4,2],200)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available:",torch.cuda.is_available())
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


def train():
	net.train()
	for batch_idx, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = net(images)                
		loss = criterion(outputs, labels)    
		loss.backward()                      
		optimizer.step()
		if batch_idx%50==0: 
			print(batch_idx,loss.item())

def eval():
	net.eval()
	test_loss = 0.0 
	correct = 0.0
	for batch_idx, (images, labels) in enumerate(val_loader):
		images = images.to(device)
		labels = labels.to(device)

		outputs = net(images)
		loss = criterion(outputs, labels)
		test_loss += loss.item()
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()
	return test_loss / len(val_loader.dataset), correct.float() / len(val_loader.dataset)

if __name__=='__main__':
	num_epochs = 1000
	for epoch in range(num_epochs):
		train()
		loss,acc = eval()
		train_scheduler.step(epoch)
		print('Epoch:%d, loss:%f, accuracy:%f' % (epoch,loss,acc))
		with open('resnet_tinyimagenet.dat', 'a') as f:
		    f.write('%d\t%f\t%f\n' % (epoch,loss,acc))
		if acc > 0.8:
			break
