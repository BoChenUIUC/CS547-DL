import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

batch_size = 128
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
        brightness=0.1*torch.randn(1),
        contrast=0.1*torch.randn(1),
        saturation=0.1*torch.randn(1),
        hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm([196,32,32])
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([196,16,16])
        self.lrelu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm([196,16,16])
        self.lrelu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm([196,8,8])
        self.lrelu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm([196,8,8])
        self.lrelu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln6 = nn.LayerNorm([196,8,8])
        self.lrelu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln7 = nn.LayerNorm([196,8,8])
        self.lrelu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln8 = nn.LayerNorm([196,4,4])
        self.lrelu8 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.ln1(self.lrelu1(self.conv1(x)))
        x = self.ln2(self.lrelu2(self.conv2(x)))
        x = self.ln3(self.lrelu3(self.conv3(x)))
        x = self.ln4(self.lrelu4(self.conv4(x)))
        x = self.ln5(self.lrelu5(self.conv5(x)))
        x = self.ln6(self.lrelu6(self.conv6(x)))
        x = self.ln7(self.lrelu7(self.conv7(x)))
        x = self.ln8(self.lrelu8(self.conv8(x)))
        x = self.pool(x)
        x = x.view(-1, 196)
        y1,y2 = self.fc1(x),self.fc10(x)
        return y1,y2

model =  discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def test():
    model.eval()
	epoch_acc = 0.0
	epoch_loss = 0.0
    epoch_counter = 0
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)
        loss = criterion(output, Y_train_batch)
	    _, predicted = torch.max(output.data, 1)
	    accuracy = predicted.eq(Y_train_batch).sum().cpu().data.numpy()
        epoch_acc += accuracy
		epoch_loss += loss.data.item()
		epoch_counter += batch_size

    epoch_acc /= epoch_counter
	epoch_loss /= (epoch_counter/batch_size)
    print("  %.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

def train(epoch):
    model.train()
    epoch_acc = 0.0
	epoch_loss = 0.0
    epoch_counter = 0
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()

        loss.backward()
        if(epoch>6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'step' in state.keys():
                        if(state['step']>=1024):
                            state['step'] = 1000
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
	    accuracy = predicted.eq(Y_train_batch).sum().cpu().data.numpy()
        epoch_acc += accuracy
		epoch_loss += loss.data.item()
		epoch_counter += batch_size
    epoch_acc /= epoch_counter
	epoch_loss /= (epoch_counter/batch_size)
    print(epoch, "  %.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)


for epoch in range(1,100):
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
	train(epoch)
	test()

torch.save(model,'cifar10.model')
