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

def resnet18(pretrained=True):
	model = torchvision.models.resnet.ResNet(
		torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
	if pretrained:
		model.load_state_dict(model_zoo.load_url(
			model_urls['resnet18'],model_dir='./'))
	return model
net = resnet18(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)
upsampler = nn.Upsample(scale_factor=7, mode='bilinear')

def train():
	net.train()
	for batch_idx, (images, labels) in enumerate(trainloader):
		images = upsampler(images)
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
		images = upsampler(images)
		images = images.to(device)
		labels = labels.to(device)

		outputs = net(images)
		loss = criterion(outputs, labels)
		test_loss += loss.item()
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()
	return test_loss / len(dataloader.dataset), correct.float() / len(dataloader.dataset)

if __name__=='__main__':
	num_epochs = 1000
	for epoch in range(num_epochs):
		train()
		test_loss,test_acc = eval(testloader)
		train_loss,train_acc  =eval(trainloader)
		train_scheduler.step(epoch)
		print('Epoch:%d, test_loss:%f, test_accuracy:%f, train_loss:%f, train_accuracy:%f' \
				% (epoch,test_loss,test_acc,train_loss,train_acc))
		with open('pretrained_cifar100.dat', 'a') as f:
		    f.write('%d\t%f\t%f\t%f\t%f\n' % (epoch,test_loss,test_acc,train_loss,train_acc))
		if test_acc > 0.75:
			break