import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim     
from torch.autograd import Variable 

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(15),
	transforms.ToTensor(),
	transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
	])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
	])

# For trainning data
trainset = torchvision.datasets.CIFAR100(root='~/scratch/.', train=True,download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=8)
# For testing data
testset = torchvision.datasets.CIFAR100(root='~/scratch/.', train=False,download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ResNet(BasicBlock,[2,4,4,2],100)
# from torchsummary import summary
# print (summary(net, (3, 32, 32)))
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20, 40, 60], gamma=0.2)

def train():
	net.train()
	for batch_idx, (images, labels) in enumerate(trainloader):
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = net(images)                
		loss = criterion(outputs, labels)    
		loss.backward()                      
		optimizer.step()
		if batch_idx%50==0: 
			print(batch_idx,loss.item())

def eval(dataloader):
	net.eval()
	test_loss = 0.0 
	correct = 0.0
	for batch_idx, (images, labels) in enumerate(dataloader):
		images = images.to(device)
		labels = labels.to(device)

		outputs = net(images)
		loss = criterion(outputs, labels)
		test_loss += loss.item()
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()
	return test_loss / len(dataloader.dataset), correct.float() / len(dataloader.dataset)

with open('resnet_cifar100.dat', 'w') as f:
	f.write('')

if __name__=='__main__':
	num_epochs = 1000
	for epoch in range(num_epochs):
		train()
		test_loss,test_acc = eval(testloader)
		train_loss,train_acc  =eval(trainloader)
		train_scheduler.step(epoch)
		print('Epoch:%d, test_loss:%f, test_accuracy:%f, train_loss:%f, train_accuracy:%f' \
				% (epoch,test_loss,test_acc,train_loss,train_acc))
		with open('resnet_cifar100.dat', 'a') as f:
		    f.write('%d\t%f\t%f\t%f\t%f\n' % (epoch,test_loss,test_acc,train_loss,train_acc))
		if test_acc > 0.65:
			break