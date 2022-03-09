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
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--nepochs", "-n", default=100, type=int, help="number of epochs")
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}


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
early_stopping = EarlyStopping(patience=20)

model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

log_title = "cifar-10-test"
log_path = "train_report/log.txt"
if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    args.checkpoint = os.path.dirname("./checkpoint/train_CIFAR10_9051_copy.pth")
    checkpoint = torch.load("./checkpoint/train_CIFAR10_9051_copy.pth")
    model.load_state_dict(checkpoint["model"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    early_stopping.best_acc = best_acc
    print(f"best_acc:", best_acc)
    logger = Logger(log_path, title=log_title, resume=True)
else:
    logger = Logger(log_path, title=log_title)
    logger.set_names(["Lr", "Train Loss", "Valid Loss", "Train Acc.", "Valid Acc."])


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
checkpoint = torch.load("./checkpoint/train_CIFAR10_9051_copy.pth")

# print(f"lr: ", optimizer.state_dict()["param_groups"][0]["lr"])

# Training
def train(epoch):
    global train_losses, train_loss, batch_idx_train, correct_train, total_train

    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for batch_idx_train, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx_train,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx_train + 1), 100.0 * correct_train / total_train, correct_train, total_train),
        )
    # append logger file
    # logger.append([train_loss / (batch_idx_train + 1)])  #  test_loss, train_acc, test_acc

    train_losses.append(train_loss)
    return train_loss


def test(epoch):
    global best_acc
    global test_losses
    global test_loss, batch_idx_test, correct_test, total_test
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_idx_test, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx_test,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx_test + 1), 100.0 * correct_test / total_test, correct_test, total_test),
            )

        test_losses.append(test_loss)

    acc = 100.0 * correct_test / total_test
    early_stopping(acc)


epoch_index = 0
while epoch_index <= 1:
    train(start_epoch + epoch_index)
    test(start_epoch + epoch_index)

    epoch_index += 1
    logger.append(
        [
            state["lr"],
            train_loss / (batch_idx_train + 1),
            test_loss / (batch_idx_test + 1),
            100.0 * correct_train / total_train,
            100.0 * correct_test / total_test,
        ]
    )
    if early_stopping.early_stop:
        break
logger.close()
logger.plot()
savefig("train_report/log.png")


# plt.plot(x, y)
fig1 = plt.figure()
plt.plot(range(epoch_index), train_losses)
plt.plot(range(epoch_index), test_losses)
plt.legend(["Train", "Validation"], prop={"size": 10})
plt.title("Loss Function", size=10)
plt.xlabel("Epoch", size=10)
plt.ylabel("Loss", size=10)
plt.ylim(ymin=0)
# plt.show()
fig1.tight_layout()
path = "train_report/train_CIFAR10_9051_copy.png"
if os.path.isfile(path):
    os.remove(path)
fig1.savefig(path)
