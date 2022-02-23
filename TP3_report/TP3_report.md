# Lab Session 2&3

**Made by:**

- Zijie NING @[mm0806son](https://github.com/mm0806son)
- Guoxiong SUN @[GuoxiongSUN](https://github.com/GuoxiongSUN)

-----

## Part 1 Pruning

### 1. VGG Global Pruning, no retrain 

We started by pruning on a VGG-11 network which is trained by us before. We observed that the accuracy dropped significantly after a certain point. We also saw a quite big occileration at around a pruning rate of 85%.

<img src="global_prunning.png" alt="global_prunning_VGG" style="zoom:80%;" />

We tried to zoom in to see this phenomenon.

<img src="noise_0.8_0.9.png" alt="noise_0.8_0.9" style="zoom:80%;" />

### 2. Densenet Global Pruning, no retrain

According to the comparaison that we mentioned last time, we dicided to change our network. Here we use DenseNet.

<img src="accuracy.png" alt="accuracy" style="zoom:80%;" />

As we had done before, we plotted the impact of pruning rate on the accuracy. We then zoomed in in order to find the best pruning rate.

<img src="CIFAR10_prun.png" alt="CIFAR10_prun" style="zoom:80%;" />

<img src="CIFAR10_prun_try_80_90.png" alt="CIFAR10_prun_try_80_90" style="zoom:80%;" />

<img src="CIFAR10_prun_try_90_95.png" alt="CIFAR10_prun_try_90_95" style="zoom:80%;" />

<img src="chosen_point_pruning.png" alt="chosen_point_pruning" style="zoom:80%;" />

By using pruning, we get a satisfying accuracy with a much smaller network.

Memory footprint:

<img src="result_memory_footprint.png" alt="result_memory_footprint" style="zoom:80%;" />


Part 2 - Reflections
--

1.An accidental attempt
<img src="result_retrain.png" alt="result_retrain" style="zoom:80%;" />


0.9235 - 80.625% -> 82.34
Number of non-zero parameters:  74151
74151*32=2372382B=2318Kb=2.26Mb

969281 88.875% -> 91.07%
969281*32=29.6Mb

2.In theory, the model should be smaller after pruning. But not yet.



Part 3 - Problems
--

1.binarization

The accuracy is reduced after binarization
(放跑完后的图)

2.For the method：[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

<img src="retrain_after_pruning.png" alt="retrain_after_pruning" style="zoom:80%;" />

how to set the weight of subsequent pruning

