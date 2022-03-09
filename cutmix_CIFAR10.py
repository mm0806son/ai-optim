"""
@Name           :cutmix_CIFAR10.py
@Description    :
@Time           :2022/03/09 10:25:05
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
import numpy as np

from models import *
from utils import progress_bar, EarlyStopping, rand_bbox

import torchvision
import torchvision.transforms as transforms

from logger import Logger, savefig

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
parser.add_argument("--beta", default=1.0, type=float, help="hyperparameter beta")
parser.add_argument("--cutmix_prob", default=1, type=float, help="cutmix probability")
parser.add_argument("--optim", default="SGD", type=str, help="optimizer: SGD, Adam")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_losses = []
test_losses = []
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

log_title = "Cifar10 CutMix"
log_path = "train_report/log.txt"
if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/cutmix_CIFAR10.pth")
    model.load_state_dict(checkpoint["model"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    early_stopping.best_acc = best_acc
    print(f"best_acc:", best_acc)
    logger = Logger(log_path, title=log_title, resume=True)
else:
    logger = Logger(log_path, title=log_title)
    # logger.set_names(["Lr", "Train Loss", "Valid Loss", "Train Acc.", "Valid Acc."])
    logger.set_names(["Train Acc.", "Valid Acc."])

criterion = nn.CrossEntropyLoss().cuda()

if args.optim == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
elif args.optim == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
else:
    raise Exception("unknown optimizer: {}".format(args.optim))


# Training
def train(epoch):
    global train_losses, train_loss_epoch, train_acc
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1.0 - lam)

        else:
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
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
    train_loss_epoch = train_loss / (batch_idx + 1)
    train_losses.append(train_loss_epoch)
    train_acc = 100.0 * correct / total


def test(epoch):
    global best_acc
    global test_losses, test_loss_epoch, test_acc
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
        test_loss_epoch = test_loss / (batch_idx + 1)
        test_losses.append(test_loss_epoch)

    test_acc = 100.0 * correct / total
    early_stopping(test_acc)

    # Save checkpoint.
    if test_acc > best_acc:
        print("Saving..")
        state = {
            "model": model.state_dict(),
            "acc": test_acc,
            "epoch": epoch,
            "loss": test_loss_epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/cutmix_CIFAR10.pth")
        best_acc = test_acc


epoch_index = 0
while epoch_index <= n_epochs:
    train(start_epoch + epoch_index)
    test(start_epoch + epoch_index)
    epoch_index += 1
    logger.append(
        [
            # optimizer.state_dict()["param_groups"][0]["lr"],
            # train_loss_epoch,
            # test_loss_epoch,
            train_acc,
            test_acc,
        ]
    )
    if early_stopping.early_stop:
        break
    if args.optim == "SGD":
        scheduler.step()

logger.close()
logger.plot()
savefig("train_report/log.png")

# # plt.plot(x, y)
# fig1 = plt.figure()
# plt.plot(range(epoch_index), train_losses)
# plt.plot(range(epoch_index), test_losses)
# plt.legend(["Train", "Validation"], prop={"size": 10})
# plt.title("Loss Function", size=10)
# plt.xlabel("Epoch", size=10)
# plt.ylabel("Loss", size=10)
# plt.ylim(ymin=0)
# # plt.show()
# fig1.tight_layout()
# path = "train_report/cutmix_CIFAR10.png"
# if os.path.isfile(path):
#     os.remove(path)
# fig1.savefig(path)
