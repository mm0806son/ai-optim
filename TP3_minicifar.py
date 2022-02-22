"""
@Name           :TP3_task1.py
@Description    :Pruning
@Time           :2022/02/16 11:19:40
@Author         :Zijie NING & Guoxiong SUN
@Version        :1.0
"""

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
"""


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
# n_epochs = 50
n_epochs = args.nepochs


# Model
print("==> Building model..")
model = VGG("VGG11")

model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/vgg11_minicifar.pth")
    model.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

parameters_to_prune = []
for module in model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, "weight"))
        # break

# print(f"parameters_to_prune:", parameters_to_prune)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

n_zeros = 0
n_elements = 0
for module in model.modules():
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

"""
print(f"m = ", module)
print(f"name_buffers:", list(module.named_buffers()))
prune.random_unstructured(module, name="weight", amount=0.3)
print(f"After pruning: ")
print(list(module.named_parameters()))
print(f"name_buffers:", list(module.named_buffers()))
print(f"module.weight:", module.weight)
# print(module._forward_pre_hooks)
prune.l1_unstructured(module, name="bias", amount=3)
print(list(module.named_parameters()))
"""

"""

# Training
def train(epoch):
    global loss_train
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
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
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
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

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/TP3.pth")
        best_acc = acc


for epoch in range(start_epoch, start_epoch + n_epochs):
    train(epoch)
    test(epoch)


# plt.plot(x, y)
fig1 = plt.figure()
plt.plot(range(n_epochs), loss_train)
plt.plot(range(n_epochs), loss_test)
plt.legend(["Train", "Validation"], prop={"size": 10})
plt.title("Loss Function", size=10)
plt.xlabel("Epoch", size=10)
plt.ylabel("Loss", size=10)
plt.ylim(ymax=20, ymin=0)
# plt.show()
fig1.tight_layout()
fig1.savefig("TP3_report/figure1.png")
"""
