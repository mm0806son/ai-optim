Report lab Session 3
--
The objectives of this third lab session is to perform experiments using pruning methods.

Part 1 Pruning
--

1. VGG Global Pruning, no retrain 

<img src="global_prunning.png" alt="global_prunning_VGG" style="zoom:80%;" />

<img src="noise_0.8_0.9.png" alt="noise_0.8_0.9" style="zoom:80%;" />

2. Densenet Global Pruning, no retrain

<img src="accuracy.png" alt="accuracy" style="zoom:80%;" />

<img src="CIFAR10_prun.png" alt="CIFAR10_prun" style="zoom:80%;" />

<img src="CIFAR10_prun_try_80_90.png" alt="CIFAR10_prun_try_80_90" style="zoom:80%;" />

<img src="CIFAR10_prun_try_90_95.png" alt="CIFAR10_prun_try_90_95" style="zoom:80%;" />

<img src="chosen_point_pruning.png" alt="chosen_point_pruning" style="zoom:80%;" />

The effect after pruning

<img src="result_memory_footprint.png" alt="result_memory_footprint" style="zoom:80%;" />





Part 2 - problem
--

1.binarization

The accuracy is reduced after binarization
(放跑完后的图)

2.For the method：[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

<img src="retrain_after_pruning.png" alt="retrain_after_pruning" style="zoom:80%;" />

how to set the weight of subsequent pruning


0.9235 - 80.625% -> 82.34
Number of non-zero parameters:  74151
74151*32=2372382B=2318Kb=2.26Mb

969281 88.875% -> 91.07%
969281*32=29.6Mb