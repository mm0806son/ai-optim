"""
@Name           :test_CIFAR10.py
@Description    :
@Time           :2022/02/16 16:09:05
@Author         :Zijie NING
@Version        :1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse

import matplotlib.pyplot as plt

from models import *
from utils import progress_bar, EarlyStopping

import torchvision
import torchvision.transforms as transforms

# Prepare Cifar10
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

"""Train CIFAR10 with PyTorch."""
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
# parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
# parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
# parser.add_argument("--nepochs", "-n", default=100, type=int, help="number of epochs")
# args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

loss_train = []
loss_test = []

# Model
print("==> Building model..")
model = densenet_cifar()

model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


# Load checkpoint.
print("==> Resuming from checkpoint..")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
checkpoint = torch.load("./checkpoint/train_CIFAR10_9051.pth")
# model.load_state_dict(checkpoint["model"])
for m in model.module():
    model.m.load_state_dict(checkpoint)
print(f"best_acc:", best_acc)


criterion = nn.CrossEntropyLoss()


def test(epoch):
    global best_acc
    global loss_test
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

        loss_test.append(test_loss)

    acc = 100.0 * correct / total


test(0)
