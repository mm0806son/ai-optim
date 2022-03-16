# Training on CIFAR10

lr=0.01 Adam 300epochs,
lr=0.001 Adam 300epochs,
lr=0.0001 Adam 300epochs

cutmix with SGD, see log_train.txt, 95.18%

We then train without cutmix, but didn't improve. OK

Tried model.half, the loss of accu is negligeble. GOOD
