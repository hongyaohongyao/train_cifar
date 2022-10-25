# [Train CIFAR10](https://github.com/hongyaohongyao/train_cifar)

This project explores the performance of some classical architecture of convolution neural networks and training methods on CIFAR10. 

Usage

- [Installation](#Installation)
- [Running Examples](#Running-Examples) 

Experiments

- [Training Different Network Architecture](#network-Architectures)
- [Deeper or Wider?](#Deeper-or-Wider) 
- [Activation Function and Layer Normalization](#Activation-Function-and-Layer-Normalization)
- [Augmentations](#Augmentations)

# Usage

## Installation

step 1. install [PyTorch](https://pytorch.org/) 

step 2. install denpendencies

```shell
pip install -r requirements [-i https://pypi.tuna.tsinghua.edu.cn/simple]
```

## Running Examples

training the resnet18 with the cosine annealing scheduler

```shell
python train.py --name <savedir> --arch resnet18 --sche cosine_annealing --lr 0.1 -e 250
```

training the desnet121 with multi-step scheduler and dropping the learning rate at epoch 20, 40, 60, 80.

```shell
python train.py --name <savedir> --arch densenet121 --lr 0.1 -m 20 -m 40 -m 60 -m 80
```

training the resnet18 with random erasing augmentation.

```shell
python train.py --name <savedir> --arch resnet18 --erase-aug
```

training the resnet18 with mixup augmentation.

```shell
python train.py --name <savedir> --arch resnet18 --mixup-aug --mixup-alpha 0.3
```

# Experiments

## Network Architectures

Training to find out which network architecture leads to better performance.

- [ResNet](https://arxiv.org/abs/1512.03385): ResNets is a classical and strong baseline in compute vision. ResNets introduce the residual structure into convolution neural network, thus solving the vanishing gradient problem and increasing the depth of CNN. Experimental results show that resnets are stable network with good performance. 

  > Residual Block
  >
  > ![image-20221024215405990](assets/README/image-20221024215405990.png)

- [VGG](https://arxiv.org/abs/1409.1556): Comparing with the previous work, AlexNets, VGG nets stacks convolution layers and pooling layers with smaller kernel size which reduce the parameters and improve classification accuracy. 

- [DenseNet](https://arxiv.org/abs/1608.06993): Similar to deep residual networks, dense networks make connections between different layers but are denser. As shown in the table below, the dense network achieves the highest accuracy on CIFAR10 at a smaller size. Practically, dense networks take up more memory at inference because the outputs of all layers are saved for concatenation.  

  > ![image-20221024214950826](assets/README/image-20221024214950826.png)

- [ResNeXt](https://arxiv.org/abs/1611.05431): ResNeXts use the Aggregated Residual to aggregated more convolution operation in one layer which is able to improve classification accuracy.

  > ![image-20221024214224765](assets/README/image-20221024214224765.png)

- [ConvMixer](https://arxiv.org/abs/2201.09792): The ConvMixer was proposed to answer a question: Is the performance of ViTs due to the inherently-more-powerful Transformer architecture, or is it at least partly due to using patches as the input representation? The network starts with a patches embeddings layer following with several DW layers. The source code training ConvMixer has reached over 95% accuracy, but it seems that more tricks are needed as it only has 93% accuracy with our training configuration.

| Model           | params(M) | Train Acc | Valid Acc |
| --------------- | --------- | --------- | --------- |
| resnet18        | 11.17     | 98.957    | 96.015    |
| vgg16           | 14.72     | 98.661    | 95.214    |
| densenet121     | 6.95      | 99.257    | 96.301    |
| resnext18       | 10.02     | 99.025    | 95.827    |
| convmixer256d16 | 1.28      | 98.649    | 93.701    |

## Deeper or Wider?

Intuitively, it's easy to improve the performance of the deep neural network by increasing the number of parameters. [Zagoruyko, S .et.al](https://arxiv.org/abs/1605.07146) show that the method to increase the parameters of the model is not only to deepen but also to widen the network. Inspired by this work, we compare the resnet34 with the modified resnets which have double channels and the same number of parameters as the resnet34. **It turns out that wider resnets lead to better performance**.

| Model          | params(M) | Train Acc | Valid Acc |
| -------------- | --------- | --------- | --------- |
| resnet18       | 11.17     | 98.957    | 96.015    |
| resnet34       | 21.28     | 98.454    | 95.659    |
| wider resnet14 | 21.06     | 99.365    | 96.539    |

## Activation Function and Layer Normalization

We trained the resnet with 4 different activation function and 4 different  layer normalization methods. It's look like that the combination of relu activation and batch normlizaion leads to a best accuracy on CIFAR10

| Model    | Activation | Normalize layer | Train Acc | Valid Acc |
| -------- | ---------- | --------------- | --------- | --------- |
| resnet18 | Relu       | Batch Norm      | 99.090    | 96.193    |
| resnet18 | Leaky Relu | Batch Norm      | 99.992    | 95.372    |
| resnet18 | Celu       | Batch Norm      | 98.143    | 91.475    |
| resnet18 | Gelu       | Batch Norm      | 99.996    | 94.709    |
| -        | -          | -               | -         | -         |
| resnet18 | Relu       | Instance Norm   | 99.996    | 95.085    |
| resnet18 | Relu       | Layer Norm      | 99.994    | 95.115    |
| resnet18 | Relu       | Group Norm      | 99.996    | 95.045    |

## Augmentations

- [Random Erasing](https://arxiv.org/abs/1708.04896v2): Erasing a rectangle area randomly from inputs, filling with the constant. As shown in the table below, random erasing reduces training accuracy while increasing validation accuracy, demonstrating that the method prevents overfitting.

  > ![image-20221025123819469](assets/README/image-20221025123819469.png)

- [Mixup](https://arxiv.org/abs/1710.09412): In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels. As shown below, mixup improves the generalization of neural network architectures.

  > ![image-20221025124317770](assets/README/image-20221025124317770.png)

| Model           | Aug       | params(M) | Train Acc | Valid Acc |
| --------------- | --------- | --------- | --------- | --------- |
| resnet18        | -         | 11.17     | 99.090    | 96.193    |
| resnet18        | Erasing   | 11.17     | 98.957    | 96.015    |
| convmixer256d16 | -         | 1.28      | 99.954    | 92.692    |
| convmixer256d16 | Erasing   | 1.28      | 98.649    | 93.701    |
| vgg16           | -         | 14.72     | 99.986    | 94.165    |
| vgg16           | Erasing   | 14.72     | 98.661    | 95.214    |
| -               | -         | -         | -         | -         |
| resnet18        | Mixup 0.2 | 11.17     | 91.378    | 95.926    |
| resnet18        | Mixup 0.3 | 11.17     | 85.472    | 95.767    |
| resnet18        | Mixup 0.4 | 11.17     | 83.565    | 96.005    |
