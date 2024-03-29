import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

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

no_of_hidden_units = 196
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm([no_of_hidden_units,32,32])
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([no_of_hidden_units,16,16])
        self.lrelu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm([no_of_hidden_units,16,16])
        self.lrelu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm([no_of_hidden_units,8,8])
        self.lrelu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm([no_of_hidden_units,8,8])
        self.lrelu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
        self.ln6 = nn.LayerNorm([no_of_hidden_units,8,8])
        self.lrelu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
        self.ln7 = nn.LayerNorm([no_of_hidden_units,8,8])
        self.lrelu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
        self.ln8 = nn.LayerNorm([no_of_hidden_units,4,4])
        self.lrelu8 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(no_of_hidden_units, 1)
        self.fc10 = nn.Linear(no_of_hidden_units, 10)

    def forward(self, x, extract_features=0):
        x = self.ln1(self.lrelu1(self.conv1(x)))
        x = self.ln2(self.lrelu2(self.conv2(x)))
        x = self.ln3(self.lrelu3(self.conv3(x)))
        x = self.ln4(self.lrelu4(self.conv4(x)))
        x = self.ln5(self.lrelu5(self.conv5(x)))
        x = self.ln6(self.lrelu6(self.conv6(x)))
        x = self.ln7(self.lrelu7(self.conv7(x)))
        x = self.ln8(self.lrelu8(self.conv8(x)))
        # if(extract_features==8):
        #     h = F.max_pool2d(x,4,4)
        #     h = h.view(-1, no_of_hidden_units)
        #     return h
        x = self.pool(x)
        x = x.view(-1, no_of_hidden_units)
        y1,y2 = self.fc1(x),self.fc10(x)
        return y1,y2

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, no_of_hidden_units*4*4)
        self.conv1 = nn.ConvTranspose2d(no_of_hidden_units, no_of_hidden_units, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv2 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv3 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv4 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv5 = nn.ConvTranspose2d(no_of_hidden_units, no_of_hidden_units, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv6 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv7 = nn.ConvTranspose2d(no_of_hidden_units, no_of_hidden_units, 4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv8 = nn.Conv2d(no_of_hidden_units, 3, 3, stride=1, padding=1)
    def forward(self,x):
        x = self.fc1(x)
        x = x.view(-1,no_of_hidden_units,4,4)
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
num_epochs = 400
import numpy as np

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

if not os.path.exists('output'):
    os.mkdir('output')

import time
start_time = time.time()

# Train the model
for epoch in range(0,num_epochs):

    # before epoch training loop starts
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []

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
            if(epoch>6):
                for group in optimizer_g.param_groups:
                    for p in group['params']:
                        state = optimizer_g.state[p]
                        if 'step' in state.keys():
                            if(state['step']>=1024):
                                state['step'] = 1000
            optimizer_g.step()

            # train D
            for p in aD.parameters():
                p.requires_grad_(True)

            aD.zero_grad()

            # train discriminator with input from generator
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()
            with torch.no_grad():
                fake_data = aG(noise)

            disc_fake_source, disc_fake_class = aD(fake_data)

            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, fake_label)

            # train discriminator with input from the discriminator
            real_data = Variable(X_train_batch).cuda()
            real_label = Variable(Y_train_batch).cuda()

            disc_real_source, disc_real_class = aD(real_data)

            prediction = disc_real_class.data.max(1)[1]
            accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, real_label)

            gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)

            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            disc_cost.backward()
            if(epoch>6):
                for group in optimizer_g.param_groups:
                    for p in group['params']:
                        state = optimizer_g.state[p]
                        if 'step' in state.keys():
                            if(state['step']>=1024):
                                state['step'] = 1000
                for group in optimizer_d.param_groups:
                    for p in group['params']:
                        state = optimizer_d.state[p]
                        if 'step' in state.keys():
                            if(state['step']>=1024):
                                state['step'] = 1000
            optimizer_d.step()

            # within the training loop
            loss1.append(gradient_penalty.item())
            loss2.append(disc_fake_source.item())
            loss3.append(disc_real_source.item())
            loss4.append(disc_real_class.item())
            loss5.append(disc_fake_class.item())
            acc1.append(accuracy)
            if((batch_idx%50)==0):
                print(epoch, batch_idx, "%.2f" % np.mean(loss1),
                                        "%.2f" % np.mean(loss2),
                                        "%.2f" % np.mean(loss3),
                                        "%.2f" % np.mean(loss4),
                                        "%.2f" % np.mean(loss5),
                                        "%.2f" % np.mean(acc1))

    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing',accuracy_test, time.time()-start_time)

    ### save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)

    if(((epoch+1)%1)==0):
        torch.save(aG,'tempG.model')
        torch.save(aD,'tempD.model')

torch.save(aG,'generator.model')
torch.save(aD,'discriminator.model')
