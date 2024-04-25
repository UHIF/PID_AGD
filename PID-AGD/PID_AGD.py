import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import torch as t
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets
import time
import matplotlib.pyplot as plt
from centerloss import CenterLoss
from torch.nn import functional as F
from datasetloadoffline import dataloadcifar
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import pytorch_moveloss
import pytorch_mcl
from torch.nn.utils import weight_norm
import torchvision.transforms as form
from IPython.display import display
import calibration_curve
import calibration_error
import PID
t.manual_seed(6)
np.random.seed(6)
t.cuda.manual_seed(6)
t.cuda.manual_seed_all(6)

start = time.time()
# Step 1 : prepare dataset

batch_size = 100
# from tensorflow.keras import datasets
# (x_train,y_tain),(x_test,y_test) = datasets.cifar10.load_data()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset=dataloadcifar.CIFAR100(
    root='/',
    train=True,
    download=False,
    transform=transform_train
)

train_dataset, val_dataset= random_split(
dataset=trainset,
lengths=[40000,10000],
generator=torch.Generator().manual_seed(1)
)

trainloader=DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

valloader=DataLoader(
    val_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

testset=dataloadcifar.CIFAR100(
    'zsy/',
    train=False,
    download=False,
    transform=transform_test
)

testloader=DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    num_workers=0
)

class Net2(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,15,kernel_size=(3,3),stride=(1,1))
        self.conv2=nn.Conv2d(15,75,kernel_size=(4,4),stride=(1,1))
        self.conv3=nn.Conv2d(75,175,kernel_size=(3,3),stride=(1,1))
        self.fc1=nn.Linear(700,200)
        self.fc2=nn.Linear(200,120)
        self.fc3=nn.Linear(120,84)
        self.fc4=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=F.max_pool2d(F.relu(self.conv3(x)),2)

        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out= self.linear(feature)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


model = ResNet34()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = pytorch_mcl.mcl(size_average=True)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)
Beta = 1.01


def train(epoch):
    running_loss = 0
    model.train()
    for batch_idx, (x, label) in enumerate(trainloader, 0):
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()

        # forward
        outputs = model(x)
        loss_cls = criterion(outputs, label, Beta)
        label = label.float()

        loss = loss_cls

        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)

        optimizer.step()

    scheduler.step()
    print("Epoch: ", epoch, "Loss is: ", loss.item())


def test(epoch):
    correct = 0
    total = 0
    model.eval()  #
    eval_loss_cls = 0
    eval_acc_cls = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_cls = criterion(outputs, labels, Beta)
            labels = labels.float()

            eval_loss_cls += loss_cls.item() * labels.size(0)
            out_argmax = torch.argmax(outputs, 1)
            eval_acc_cls += (out_argmax == labels).sum().item()

            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += torch.eq(pred, labels).float().sum().item()
    print("Epoch", epoch, "Accuracy on test set: %d %%" % (100 * correct / total))
    return correct / total


if __name__ == "__main__":
    epoch_list = []
    acc_list = []
    ece_val_list = []
    mse_val_list = []
    ece_test_list = []
    mse_test_list = []
    r_ece_list = []
    Beta_list = []
    for epoch in range(200):
        train(epoch)
        acc = test(epoch)
        acc_list.append(acc)
        rce, ece, mse = calibration_error.calibration_error(model, testloader)
        r_ece_list.append(rce)
        print(ece)
        ece_test_list.append(ece)
        mse_test_list.append(mse)
        rce, ece, mse = calibration_error.calibration_error(model, valloader)
        print(ece)
        ece_val_list.append(ece)
        mse_val_list.append(mse)

        if epoch % 1 == 0:
            PID_controller = PID.PID(P=1, I=1, D=1)
            Beta_change = PID_controller.control(rce)
            Beta_before = Beta
            Beta= Beta*math.exp(-Beta_change)
            print(Beta_change)
            print(Beta)
            print(rce)
            Beta_list.append(Beta)

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (Beta_before * math.log(Beta_before) - Beta_before + 1) / (
                            (Beta_before - 1) * (Beta_before - 1)) * (Beta - 1) * (Beta - 1) / (
                                                Beta * math.log(Beta) - Beta + 1)

        with open('/', 'w') as f:
            f.write(str(acc_list) + '\n')
            f.write(str(r_ece_list) + '\n')
            f.write(str(ece_val_list) + '\n')
            f.write(str(mse_val_list) + '\n')
            f.write(str(ece_test_list) + '\n')
            f.write(str(mse_test_list) + '\n')
            f.write(str(Beta_list) + '\n')


    ResNet34_original = model
    torch.save(ResNet34_original, '/')

    end = time.time()
    print("Total Time: ", end - start)
