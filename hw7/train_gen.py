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

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 196*4*4)
        self.conv1 = nn.ConvTranspose2d(196, 196, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(196)

        self.conv2 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(196)

        self.conv3 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(196)

        self.conv4 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(196)

        self.conv5 = nn.ConvTranspose2d(196, 196, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(196)

        self.conv6 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(196)

        self.conv7 = nn.ConvTranspose2d(196, 196, 4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(196)

        self.conv8 = nn.Conv2d(196, 3, 3, stride=1, padding=1)
    def forward(self,x):
        x = self.fc1(x)
        x = x.view(-1,196,4,4)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.conv8(x)
        return torch.tanh(x)

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

aD =  discriminator()
aD.cuda()

aG = generator()
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()

n_z = 100
n_classes = 10
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()

gen_train = 5

start_time = time.time()

# Train the model
for epoch in range(0,num_epochs):

    aG.train()
    aD.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue
        # train G
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            optimizer_g.step()
