import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):

    def __init__(self,
                 in_planes,
                 growth_rate,
                 activation=None,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.bn1 = norm_layer(in_planes)
        self.conv1 = nn.Conv2d(in_planes,
                               4 * growth_rate,
                               kernel_size=1,
                               bias=False)
        self.act1 = activation()
        self.bn2 = norm_layer(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.act2 = activation()

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 activation=None,
                 norm_layer=None):
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(Transition, self).__init__()
        self.act = activation()
        self.bn = norm_layer(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(self.act(self.bn(x)))
        out = self.avg_pool(out)
        return out


class DenseNet(nn.Module):

    def __init__(self,
                 block,
                 nblocks,
                 growth_rate=12,
                 reduction=0.5,
                 num_classes=10,
                 activation=None,
                 norm_layer=None):
        super(DenseNet, self).__init__()
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3,
                               num_planes,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.act = activation()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.bn = norm_layer(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self,
                           block,
                           in_planes,
                           nblock,
                           activation=None,
                           norm_layer=None):
        layers = []
        for i in range(nblock):
            layers.append(
                block(in_planes,
                      self.growth_rate,
                      activation=activation,
                      norm_layer=norm_layer))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.avg_pool(self.act(self.bn(out)))
        out = self.flatten(out)
        latent = out.clone()
        out = self.linear(out)
        if with_latent:
            return out, latent
        return out


def densenet121(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, **kwargs)


def densenet169(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, **kwargs)


def densenet201(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, **kwargs)


def densenet161(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, **kwargs)


if __name__ == '__main__':
    """
    """
    net = densenet121()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)
