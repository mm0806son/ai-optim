"""
@Name           :TP3_task1.py
@Description    :Pruning
@Time           :2022/02/16 11:19:40
@Author         :Zijie NING & Guoxiong SUN
@Version        :1.0
"""

# Data
print("==> Preparing data..")

from minicifar import minicifar_train, minicifar_test, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train, batch_size=800, sampler=train_sampler)
validloader = DataLoader(minicifar_train, batch_size=800, sampler=valid_sampler)
testloader = DataLoader(minicifar_test, batch_size=800)

import torchvision
import torchvision.transforms as transforms

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


"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import copy

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--nepochs", "-n", default=100, type=int, help="number of epochs")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

loss_train = []
loss_test = []
accu_test = []
# n_epochs = 50
n_epochs = args.nepochs


# Model
print("==> Building model..")
model = VGG("VGG11")

model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Load checkpoint.
print("==> Resuming from checkpoint..")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
checkpoint = torch.load("./checkpoint/vgg11_minicifar.pth")
model.load_state_dict(checkpoint["net"])
best_acc = checkpoint["acc"]
start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def test(epoch):
    global best_acc
    global loss_test
    model_pruned.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_pruned(inputs)
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
    acc = 100.0 * correct / total
    loss_test.append(test_loss)
    accu_test.append(acc)


fig1 = plt.figure()

amount_list = np.arange(0.0, 0.11, 0.1)
for amount in amount_list:
    parameters_to_prune = []
    model_pruned = copy.deepcopy(model)
    for module in model_pruned.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    # print(f"parameters_to_prune:", parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    n_zeros = 0
    n_elements = 0
    for module in model_pruned.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            n_zeros += torch.sum(module.weight == 0)
            n_elements += module.weight.nelement()
            print(
                f"Sparsity in ",
                module,
                ": ",
                100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()),
                " %",
            )

    print(f"Global sparsity: ", 100.0 * float(n_zeros) / float(n_elements))
    print(f"Number of non-zero parameters: ", int(n_elements) - int(n_zeros))
    test(0)

plt.plot(amount_list, accu_test, label=f"Accuracy")

plt.legend()  # prop={"size": 10}
plt.title("Global Pruning", size=10)
plt.xlabel("Amount", size=10)
plt.ylabel("Accuracy", size=10)
plt.ylim(ymin=0, ymax=100)
# plt.show()
fig1.tight_layout()
fig1.savefig("TP3_report/figure1.png")
