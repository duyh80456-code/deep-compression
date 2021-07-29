# [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

[![Total alerts](https://img.shields.io/lgtm/alerts/g/jack-willturner/DeepCompression-PyTorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jack-willturner/DeepCompression-PyTorch/alerts/) 
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jack-willturner/DeepCompression-PyTorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jack-willturner/DeepCompression-PyTorch/context:python)
![GitHub](https://img.shields.io/github/license/jack-willturner/DeepCompression-PyTorch)

A PyTorch implementation of [this paper](https://arxiv.org/abs/1506.02626).

To run, try:
```bash
python train.py --model='resnet34' --checkpoint='resnet34'
python prune.py --model='resnet34' --checkpoint='resnet34'
```

## Bring your own models 
In order to add a new model family to the repository you basically just need to do two things:
1. Swap out the convolutional layers to use the `ConvBNReLU` class
2. Define a `get_prunable_layers` method which returns all the instances of `ConvBNReLU` which you want to be prunable

## Summary

Given a family of ResNets, we can construct a Pareto frontier of the tradeoff between accuracy and number of parameters:

![alt text](./resources/resnets.png)

Han et al. posit that we can beat this Pareto frontier by leaving network structures fixed, but removing individual parameters:

![alt text](./resources/pareto.png)
