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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

  

class Net(nn.Module):                 
    def __init__(self):    
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0) 
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)  

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)  

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 3 * 3, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
 
    def forward(self, x):                 
        x = self.drop(self.bn1(F.relu(self.conv1(x)))) 				# 32->30
        x = self.pool(self.bn2(F.relu(self.conv2(x))))			 	# 30->28->14
        x = self.drop(self.bn3(F.relu(self.conv3(x)))) 				# 14->12
        x = self.pool(self.bn4(F.relu(self.conv4(x)))) 				# 12->10->5
        x = self.drop(self.bn5(F.relu(self.conv5(x)))) 				# 5->3
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
 
criterion = nn.CrossEntropyLoss()    
optimizer = optim.Adam(net.parameters(), lr=0.001)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

def test():
	correct = 0.0   
	total = 0.0     
	for data in testloader: 
	    images, labels = data[0].to(device), data[1].to(device)
	    outputs = net(images)
	    _, predicted = torch.max(outputs.data, 1)
	    total += labels.size(0)
	    correct += (predicted == labels).sum()

	return correct.item() / total

def train():
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data[0].to(device), data[1].to(device)

		optimizer.zero_grad()                
     
		outputs = net(inputs)                
		loss = criterion(outputs, labels)    
		loss.backward()                      
		optimizer.step()

for epoch in range(1,1000): 
	train()
	acc = test()

	print('Epoch:%d, accuracy:%f' % (epoch,acc))
	with open('output.txt', 'a') as f:
	    f.write('Epoch:%d, accuracy:%f\n' % (epoch,acc))
	if acc>0.85:
		break
 
print('Finished Training')

       


   