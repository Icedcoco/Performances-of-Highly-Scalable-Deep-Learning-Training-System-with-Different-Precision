'''Train CIFAR10 with PyTorch.'''
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


import sys
import time
import os
import argparse
import math


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=1000, type=int, help='compute epochs')
parser.add_argument('--use_cuda', default=False, action='store_true', help='use gpu')
parser.add_argument('--data_path', default='./data', type=str, help='root path of data')
parser.add_argument('--num_workers', default=2, type=int, help='number of data loader workers')
parser.add_argument('--local_rank', default=-1, type=int, help='local device rank')
parser.add_argument('--optimizer', default='sgd', type=str, help='sgd, sgdn, adagrad, adadelta or adam')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--use_amp', action='store_true', help='use Automatic Mixed Precision')
parser.add_argument('--use_half', action='store_true', help='use FP16')
parser.add_argument('--use_34', action='store_true', help='use ResNet-34')
parser.add_argument('--use_50', action='store_true', help='use ResNet-50')
parser.add_argument('--use_ddp', action='store_true', help='use Distributed Data Parallel')
args = parser.parse_args()

print('cuda:')
print(torch.cuda.is_available())
if args.use_cuda:
    device = 'cuda'
else:
    device = 'cpu'
best_acc = 0  # best accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
compute_epoch = args.epoch

if args.use_ddp:
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)
if args.use_ddp:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                num_workers=args.num_workers, sampler=DistributedSampler(trainset))

testset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
        num_workers=args.num_workers)
if args.use_ddp:
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            num_workers=args.num_workers, sampler=DistributedSampler(testset))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

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
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

print('==> Building model..')
net = ResNet18()
if args.use_34:
    net = ResNet34()
if args.use_50:
    net = ResNet50()
net = net.to(device)
if args.use_half:
    net = net.half()
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
if args.use_ddp:
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
elif device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint_amp'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_amp/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
if args.optimizer == 'sgdn':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4, nesterov=True)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr,
                              weight_decay=5e-4)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr,
                               weight_decay=5e-4)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0000000001)
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('The size of the model is: {:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

# Training
def train(epoch):
    flag = True
    net.train()
    train_loss = 0
    total_correct = 0
    total = 0
    total_training_time = 0
    total_computing_time = 0
    total_data_time = 0
    total_move_time = 0
    time1 = torch.cuda.Event(enable_timing=True)
    time2 = torch.cuda.Event(enable_timing=True)
    time3 = torch.cuda.Event(enable_timing=True)
    time4 = torch.cuda.Event(enable_timing=True)
    time1.record()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        time2.record()
        # mid_training_time = time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        if args.use_half:
            inputs = inputs.half()
        time3.record()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        time4.record()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct = predicted.eq(targets).sum().item()
        total_correct += correct

        torch.cuda.synchronize()

        total_computing_time  += time3.elapsed_time(time4)
        total_data_time  += time1.elapsed_time(time2)
        total_move_time  += time2.elapsed_time(time3)
        total_training_time += time1.elapsed_time(time4)

        time1.record()
    avg_loss = train_loss/total
    acc = 100.*total_correct/total
    if args.use_ddp:
        print('Rank %d: Training epoch %d: Loss=%.3f Acc=%.3f Training_time: %.2fs' % 
            (local_rank, epoch, avg_loss, acc, total_training_time/1000))
        print('Rank %d: Training time stat of epoch %d: compute=%.3fs data=%.3fs move=%.3fs' % 
            (local_rank, epoch, total_computing_time/1000, total_data_time/1000, total_move_time/1000))
    else:
        print('Training epoch %d: Loss=%.3f Acc=%.3f Training_time: %.2fs' % 
              (epoch, avg_loss, acc, total_training_time/1000))
        print('Training time stat of epoch %d: compute=%.3fs data=%.3fs move=%.3fs' % 
              (epoch, total_computing_time/1000, total_data_time/1000, total_move_time/1000))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_testing_time = 0
    time1 = torch.cuda.Event(enable_timing=True)
    time2 = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        time1.record()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.use_half:
                inputs = inputs.half()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            time2.record()

            test_loss += (loss.item() * targets.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            torch.cuda.synchronize()
            total_testing_time += time1.elapsed_time(time2)
            time1.record()

    # Save checkpoint.
    avg_loss = test_loss/total
    acc = 100.*correct/total
    if args.use_ddp:
        print('Rank %d: Testing epoch %d: Loss=%.3f Acc=%.3f Testing_time: %.2fs tota=%d' % 
            (local_rank, epoch, avg_loss, acc, total_testing_time/1000, total))
    else:
        print('Testing epoch %d: Loss=%.3f Acc=%.3f Testing_time: %.2fs tota=%d' % 
            (epoch, avg_loss, acc, total_testing_time/1000, total))
          
'''
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_amp_005_4c_2w_128'):
            os.mkdir('checkpoint_amp_005_4c_2w_128')
        torch.save(state, './checkpoint_amp_005_4c_2w_128/ckpt.pth')
        best_acc = acc
'''

time_sum = 0
getModelSize(net)
Start_time = time.perf_counter()
for epoch in range(start_epoch, start_epoch+compute_epoch):
    trainloader.sampler.set_epoch(epoch)
    testloader.sampler.set_epoch(epoch)
    time1 = time.perf_counter()
    train(epoch)
    # if epoch%5 == 0:
    #     test(epoch)
    time2 = time.perf_counter()
    test(epoch)
    Cur_time = time.perf_counter()
    scheduler.step()
    if args.use_ddp:
        print('Rank %d: Epoch %d: %.2fs' % (local_rank, epoch, Cur_time - Start_time))
        print('Rank %d: Total Train: %.2fs, Test: %.2fs' % (local_rank, time2 - time1, Cur_time - time2))
    else:
        print('Epoch %d: %.2fs' % (epoch, Cur_time - Start_time))
        print('Total Train: %.2fs, Test: %.2fs' % (time2 - time1, Cur_time - time2))
    sys.stdout.flush()
getModelSize(net)
print("end")
print(" ")
print(" ")
