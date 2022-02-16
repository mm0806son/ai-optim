"""
@Name           :TP3_try_pruning.py
@Description    :Pruning
@Time           :2022/02/16 11:19:40
@Author         :Zijie NING & Guoxiong SUN
@Version        :1.0
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
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
