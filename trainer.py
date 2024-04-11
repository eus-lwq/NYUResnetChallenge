# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jwgn8r6TrNPgZh5uX8xMSCaLzhlsHHRx

Citation:

https://colab.research.google.com/github/Rakshit-Shetty/Resnet-Implementation/blob/master/ResNet_Implementation_on_CIFAR10.ipynb

Reference:
https://github.com/kuangliu/pytorch-cifar?tab=readme-ov-file
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

# CIFAR10 Images Classification with Resnet-18

## Library Import
"""

# Commented out IPython magic to ensure Python compatibility.
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
# from pytorch_model_summary import summary
from tap import Tap
from ast import literal_eval
import wandb
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import csv
import sys
import time
import math
import shutil
import matplotlib.pyplot as plt
# from PIL import Image
# %matplotlib inline
from data_loader import load_cifar_batch, load_cifar_test_batch, load_cifar_test_nolabels, CIFAR10Dataset
from utils import count_all_parameters, count_parameters

##################### Args #######################
class ArgsParser(Tap):
    # hyperparameters
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    n_epochs: int = 100
    start_epoch: int = 0
    best_acc: float = 0
    batch_size: int = 20
    num_workers: int = 0
    valid_size: float = 0.2
    optimizer: str = "SGD" # Optimizer type
    self_supervise: bool = False
    # data paths
    test_data_path: str = '/home/yw7486/vast_home/Courses/dl-7123/deep-learning-mini-project-spring-24-nyu/cifar-10-python/cifar-10-batches-py/test_batch'
    test_nolabel_path: str = '/home/yw7486/vast_home/Courses/dl-7123/deep-learning-mini-project-spring-24-nyu/cifar_test_nolabels.pkl'
    data_dir: str = '/home/yw7486/vast_home/Courses/dl-7123/deep-learning-mini-project-spring-24-nyu/cifar-10-python/cifar-10-batches-py/'
    # architectural modifications
    model_arch: str = 'custom'
    stride: int = 1  
    in_planes: int = 64  
    num_channels: str = "64,128,256,512" 
    filter_sizes: str = "3,3,3,3" 
    num_blocks: str = "1,1,1,1" 
    # kernel_sizes: list = [1, 1, 1, 1]  # Kernel sizes for skip connections
    # avgpool_size: int = 4  # Pool size for the average pool layer

    def configure(self):
        self.description = 'Argument parser for training different ResNet models on CIFAR-10'

    def process_args(self):
        self.num_blocks = list(map(int, self.num_blocks.split(',')))
        self.num_channels = list(map(int, self.num_channels.split(',')))
        self.filter_sizes = list(map(int, self.filter_sizes.split(',')))

args = ArgsParser().parse_args() # Parse the arguments
def get_resnet_model(arch: str):
    if arch == 'resnet18':
        model = ResNet18()
    elif arch == 'resnet34':
        model = ResNet34()
    elif arch == 'custom':
        model = ResNet_custom()
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    return model

def get_optimizer(optimizer:str):
    # Choose the optimizer based on the optimizer argument
    if optimizer.lower() == 'sgd':
        return optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif optimizer.lower() == 'adam':
        return optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer}')

def initialize_wandb(args):
    wandb.init(project='NYUcifarChallenge', entity='eustinalwq')
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "model_architecture": args.model_arch,
        "optimizer": args.optimizer,
    }
    print("Args: ", args)

def check_cuda():
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU')
        device = torch.device("cpu")
    else:
        print('CUDA is available!  Training on GPU')
        device = torch.device("cuda")
    return train_on_gpu, device

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=args.stride):
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

    def __init__(self, in_planes, planes, stride=args.stride):
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
    def __init__(self, block, num_blocks=args.num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        print("args.num_blocks in resnet",num_blocks)
        self.in_planes = args.in_planes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, args.num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, args.num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, args.num_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, args.num_channels[3], num_blocks[3], stride=2)
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
    
def ResNet_custom():
    return ResNet(BasicBlock, args.num_blocks)
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def trainer(args, train_on_gpu, train_loader, valid_loader, net, criterion, optimizer, scheduler):
    """### Training Start"""
    # number of epochs to train the model
    n_epochs = args.n_epochs
    valid_loss_min = np.Inf # track change in validation loss
    train_losses = []
    valid_losses = []

    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            wandb.log({"train_loss": loss.item()})
        ######################
        # validate the model #
        ######################
        net.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            if train_on_gpu:
              data, target = data.cuda(), target.cuda()
            output = net(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            wandb.log({"valid_loss": loss.item()})
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # Update learning rate with scheduler
        scheduler.step()
        # Optional: Log learning rate to wandb or print it out
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: Current learning rate: {current_lr}")
        wandb.log({'epoch': epoch, 'lr': current_lr})

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
            torch.save(net.state_dict(), 'ResNet18.pt')
            wandb.save('ResNet18.pt')
            valid_loss_min = valid_loss

    net.load_state_dict(torch.load('ResNet18.pt'))

def tester_nolabel(args, train_on_gpu, num_workers, batch_size, transform_test, net):
    """
    Function to test the model on the professor's test file without labels, 
    and save the predictions to a CSV file.
    """
    test_data_path = args.test_nolabel_path
    # Loading the test batch without labels using the full path
    test_data = load_cifar_test_nolabels(test_data_path)
    test_dataset = CIFAR10Dataset(test_data, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # Ensure the network is in evaluation mode
    net.eval()

    # Predictions list
    predictions = []

    # Iterate over test data
    for batch_idx, data in enumerate(test_loader):
    # Move tensors to GPU if CUDA is available
        if train_on_gpu:
            data = data.cuda()
        output = net(data)
        _, pred = torch.max(output, 1)
        predictions.append(pred.cpu().numpy()) 

        # Concatenate all the predictions into one array
    predictions = np.concatenate(predictions, axis=0)

    # Path to the CSV output file
    output_file = 'submission.csv'

    # Writing to CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Labels'])  # Writing the header
        for idx, label in enumerate(predictions):
            writer.writerow([idx, label])  # Writing each ID and its corresponding label

    print(f"Submission file saved to {output_file}")
    # Save submission.csv to wandb
    wandb.save('submission.csv')
    print("Submission file saved to wandb")

def self_supervise_learning(args, train_on_gpu, num_workers, batch_size, train_sampler, transform_train, transform_test, train_dataset, valid_loader, net, criterion, optimizer, scheduler):
    """
    self supervised learning 
    add pseudo labels to the training data and train the model again
    """
    print("ended first training, starting self supervised learning...")
    # Load the unlabeled data and make predictions
    unlabeled_data = load_cifar_test_nolabels(args.test_nolabel_path)
    unlabeled_dataset = CIFAR10Dataset(unlabeled_data, transform=transform_test)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, num_workers=num_workers)

    # Make predictions on the unlabeled data
    net.eval()
    unlabeled_predictions = []
    for batch_idx, data in enumerate(unlabeled_loader):
        if train_on_gpu:
            data = data.cuda()
        output = net(data)
        _, pred = torch.max(output, 1)
        unlabeled_predictions.append(pred.cpu().numpy())
    unlabeled_predictions = np.concatenate(unlabeled_predictions, axis=0)

    # Create a new dataset using these predictions as labels and add it to the training data
    pseudo_labeled_dataset = CIFAR10Dataset(unlabeled_data, unlabeled_predictions, transform=transform_train)

    # Concatenate the pseudo-labeled dataset with the original training dataset
    combined_train_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_labeled_dataset])

    # Update the train loader
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)

    # Train the model with the pseudo-labeled data
    trainer(args, train_on_gpu, train_loader, valid_loader, net, criterion, optimizer, scheduler)
    print("ended self supervised learning")

def tester_withlabel(args, train_on_gpu, num_workers, batch_size, transform_test, classes, net, criterion):
    """
    Test the model on the test data with labels, 
    showing the accuracy for each class and overall
    """

    # Load the test data
    test_data, test_labels = load_cifar_test_batch(args.test_data_path)
    if test_labels is not None:
        test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform_test)
    else:
        test_dataset = CIFAR10Dataset(test_data, transform=transform_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    net.eval()
    # iterate over test data
    for batch_idx, (data, target) in enumerate(test_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        # for i in range(batch_size):
        for i in range(data.size(0)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    
    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
      if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
        classes[i], 100 * class_correct[i] / class_total[i],
        np.sum(class_correct[i]), np.sum(class_total[i])))
      else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


##################### Main #######################
initialize_wandb(args) # Initialize wandb
train_on_gpu, device = check_cuda() # Check if CUDA is available

##################### Data Loading #######################
data_dir = args.data_dir
batch_files = [os.path.join(data_dir, 'data_batch_{}'.format(i)) for i in range(1, 6)]
data_batches = [load_cifar_batch(batch_file) for batch_file in batch_files]
train_data = np.vstack([batch[0] for batch in data_batches])
train_labels = np.hstack([batch[1] for batch in data_batches])

##################### Data Augmentation #######################
# number of subprocesses to use for data loading
num_workers = args.num_workers
batch_size = args.batch_size
valid_size = args.valid_size
# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
# Generate new validation indices starting from 0
valid_idx = list(range(0, split))
# Make sure the new indices are within the bounds
assert min(valid_idx) >= 0 and max(valid_idx) < 10000, "Valid indices are out of bounds."
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
# Define transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=0),
    transforms.resize(32),
    transforms.CenterCrop(28),
    transforms.resize(32),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # new aug
    transforms.RandomRotation(degrees=15),# new aug
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Create dataset objects
train_dataset = CIFAR10Dataset(train_data, train_labels, transform=transform_train)
valid_dataset = CIFAR10Dataset(train_data[valid_idx], train_labels[valid_idx], transform=transform_test)
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, drop_last=True)
# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

##################### Model Define #################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
best_acc = args.best_acc  # best test accuracy
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
# epoch = 200
# Model
print('==> Building model..')
# net = ResNet18()
net = get_resnet_model(args.model_arch)
net = net.to(device)
total_params = count_all_parameters(net)
print(f"The model has {total_params:,} total parameters, including non-trainable ones.")
# Now count the parameters:
total_params = count_parameters(net)
print(f"The model has {total_params:,} trainable parameters.")
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(args.optimizer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
trainer(args, train_on_gpu, train_loader, valid_loader, net, criterion, optimizer, scheduler)
if args.self_supervise: # if self supervised learning is enabled then run this block
    self_supervise_learning(args, train_on_gpu, num_workers, batch_size, train_sampler, transform_train, transform_test, train_dataset, valid_loader, net, criterion, optimizer, scheduler)
tester_nolabel(args, train_on_gpu, num_workers, batch_size, transform_test, net)
tester_withlabel(args, train_on_gpu, num_workers, batch_size, transform_test, classes, net, criterion)
