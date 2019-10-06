import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim     
from torch.autograd import Variable 

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# For trainning data
trainset = torchvision.datasets.CIFAR100(root='~/scratch/.', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=8)
# For testing data
testset = torchvision.datasets.CIFAR100(root='~/scratch/.', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

num_epochs = 1000


def train():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	for epoch in range(num_epochs):
		for batch_idx, (images, labels) in enumerate(trainloader):
			images = images.to(device)
			labels = labels.to(device)