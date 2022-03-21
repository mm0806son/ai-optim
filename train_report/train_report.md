# Training on CIFAR10

lr=0.01 Adam 300epochs,
lr=0.001 Adam 300epochs,
lr=0.0001 Adam 300epochs

cutmix with SGD, see log_cutmix.txt, 95.18%

We then train without cutmix, but didn't improve. OK

Tried model.half, the loss of accu is negligeble. GOOD

For `densenet_tiny`
Flops: 49598008.0, Params: 129557.0
Score flops: 0.059444168944812116 Score Params: 0.023189089062590332
Final score: 0.08263325800740245

For `densenet_cifar`
Flops: 192701552.0, Params: 500309.0
Score flops: 0.23095652577449274 Score Params: 0.08954907847368732
Final score: 0.3205056042481801

For `densenet_small`
Flops: 109461592.0, Params: 284783.0
Score flops: 0.13119182866812099 Score Params: 0.05097260935736134
Final score: 0.18216443802548232

def densenet_small():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=9)
lr=0.1, 300epochs, 94.1%