This repo is an attempt to implement sync bn in official deeplab. It is still under constructing and do not use it.

official deeplab:
https://github.com/tensorflow/models/tree/master/research/deeplab

My implementation is inspired by
https://github.com/twke18/Adaptive_Affinity_Fields/blob/master/network/multigpu/layers.py

https://github.com/holyseven/PSPNet-TF-Reproduce/blob/master/model/utils_mg.py

Currently this solution seems not working and I am trying to go with another style of solutions, which use tf.contrib.nccl library as the repos shown below.

https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/batch_norm.py

https://github.com/jianlong-yuan/syncbn-tensorflow
