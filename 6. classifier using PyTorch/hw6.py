import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class ResBlock(nn.Module):
  def __init__(self, in_channel, out_channel, downsample=False):
    super(ResBlock, self).__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.downsample = downsample
    self.build_layer()

  def build_layer(self):
    self.batchNorm_in = nn.BatchNorm2d(self.in_channel,)
    self.relu = nn.ReLU()
    self.conv_1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
    self.conv_downsample = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=2, padding=1)
    self.conv_pool = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=2)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.batchNorm_out = nn.BatchNorm2d(self.out_channel)
    self.conv_2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

  def residual(self, x):
    x = self.batchNorm_in(x)
    x = self.relu(x)
    if self.downsample:
      x = self.conv_downsample(x)
    else:
      x = self.conv_1(x)
    x = self.batchNorm_out(x)
    x = self.conv_2(x)
    return x
  
  def shortcut(self, x):
    if self.downsample:
      x = self.batchNorm_in(x)
      x = self.relu(x)
      ## Maxpooling
      x = self.maxpool(x)
      y = torch.zeros_like(x)
      x = torch.cat([x,y],1)
      ## 1x1 Conv 
      # x = self.conv_pool(x)
    return x

  def forward(self, x):
    return self.residual(x) + self.shortcut(x)

class IdentityResNet(nn.Module):
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
        self.n1 = nblk_stage1
        self.n2 = nblk_stage2
        self.n3 = nblk_stage3
        self.n4 = nblk_stage4
        self.build_layer()

    def build_layer(self):
        layers = (nn.ModuleList() for i in range(4))
        self.layers_stage1, self.layers_stage2, self.layers_stage3, self.layers_stage4 = layers
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        ## stage1_blocks
        for _ in range(self.n1):
          self.layers_stage1.append(ResBlock(64, 64))
        ## stage2_blocks
        self.layers_stage2.append(ResBlock(64,128, downsample=True))
        for _ in range(self.n2-1):
          self.layers_stage2.append(ResBlock(128,128))
        ## stage3_blocks
        self.layers_stage3.append(ResBlock(128,256, downsample=True))
        for _ in range(self.n3-1):
          self.layers_stage3.append(ResBlock(256,256))
        ## stage4_blocks
        self.layers_stage4.append(ResBlock(256,512, downsample=True))
        for _ in range(self.n4-1):
          self.layers_stage4.append(ResBlock(512,512))
        
        self.FCLayer = nn.Linear(512, 10)

    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################

    ########################################
    # You can define whatever methods
    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################
        # 3x3 conv
        x = self.conv(x) #
        # First stage
        for layer in self.layers_stage1:
          x = layer(x)
        # Second stage
        for layer in self.layers_stage2:
          x = layer(x)
        # Third stage
        for layer in self.layers_stage3:
          x = layer(x)
        # Fourth stage
        for layer in self.layers_stage4:
          x = layer(x)
        # Average Pooling
        out = F.avg_pool2d(x, 4)
        # FC Layer
        out = out.reshape([-1,512])
        out = self.FCLayer(out)
        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 4

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net = net.to(dev)

# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()

        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)
        
        # set loss
        loss = criterion(outputs, labels)

        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')

# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')