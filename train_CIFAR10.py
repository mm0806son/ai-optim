"""
@Name           :train_CIFAR10.py
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
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--nepochs", "-n", default=300, type=int, help="number of epochs")
parser.add_argument("--optim", default="SGD", type=str, help="optimizer: SGD, Adam")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

loss_train = []
loss_test = []
# n_epochs = 50
n_epochs = args.nepochs


# Model
print("==> Building model..")
model = densenet_cifar()

# early stop
print("INFO: Initializing early stopping")
early_stopping = EarlyStopping(patience=200)

model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/train_CIFAR10_9051.pth")
    model.load_state_dict(checkpoint["model"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    early_stopping.best_acc = best_acc
    print(f"best_acc:", best_acc)


criterion = nn.CrossEntropyLoss()

if args.optim == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
elif args.optim == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
else:
    raise Exception("unknown optimizer: {}".format(args.optim))

# Training
def train(epoch):
    global loss_train
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    loss_train.append(train_loss)


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
    early_stopping(acc)

    # Save checkpoint.
    if acc > best_acc:
        print("Saving..")
        state = {
            "model": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "loss": test_loss,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/train_CIFAR10.pth")
        best_acc = acc


epoch_index = 0
while epoch_index <= n_epochs:
    train(start_epoch + epoch_index)
    test(start_epoch + epoch_index)
    epoch_index += 1
    if early_stopping.early_stop:
        break
    if args.optim == "SGD":
        scheduler.step()


# plt.plot(x, y)
fig1 = plt.figure()
plt.plot(range(epoch_index), loss_train)
plt.plot(range(epoch_index), loss_test)
plt.legend(["Train", "Validation"], prop={"size": 10})
plt.title("Loss Function", size=10)
plt.xlabel("Epoch", size=10)
plt.ylabel("Loss", size=10)
plt.ylim(ymin=0)
# plt.show()
fig1.tight_layout()
path = "train_report/train_CIFAR10.png"
if os.path.isfile(path):
    os.remove(path)
fig1.savefig(path)
