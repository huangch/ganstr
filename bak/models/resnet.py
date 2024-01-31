import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Generator(nn.Module):
    # def __init__(self, block, num_blocks, num_classes=10):
        
    def __init__(self, gene_num, latent_dim, block=BasicBlock, num_blocks=[18, 18, 18, 18]):


        super(Generator, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128, gene_num)
        self.activate = nn.Softmax(dim=1)  
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.activate(out)
        return out

# def resnet142(num_classes):
#     return ResNet(BasicBlock, [18, 18, 18, 18], num_classes)



class Discriminator(nn.Module):
    def __init__(self, gene_num, subtype_num):
        super(Discriminator, self).__init__()

        self.fcnLayer = nn.Sequential(
            nn.BatchNorm1d(gene_num),
            nn.Linear(gene_num, 1024),
            nn.LeakyReLU(0.1, inplace=True), # nn.ReLU(inplace=True), # nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25), 
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1, inplace=True), # nn.ReLU(inplace=True), # nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25),
            )

        # Output layers
        self.advLayer = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024, 1), nn.Sigmoid())
        self.auxLayer = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024, subtype_num+1), nn.Softmax(dim=1))
        self.clsLayer = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024, subtype_num), nn.Softmax(dim=1))
        
    def forward(self, trns):
        output = trns
        output = self.fcnLayer(output)
        validity = self.advLayer(output)
        label = self.auxLayer(output)
        cls = self.clsLayer(output)

        return validity, label,cls
    