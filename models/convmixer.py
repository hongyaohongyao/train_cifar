# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch.nn as nn


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim,
              depth,
              kernel_size=7,
              patch_size=2,
              num_classes=10,
              activation=None,
              norm_layer=None):
    if activation is None:
        activation = lambda: nn.GELU()
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        activation(), norm_layer(dim), *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(dim,
                                  dim,
                                  kernel_size,
                                  groups=dim,
                                  padding=kernel_size // 2), activation(),
                        norm_layer(dim))), nn.Conv2d(dim, dim, kernel_size=1),
                activation(), norm_layer(dim)) for i in range(depth)
        ], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.Linear(dim, num_classes))


def convmixer256d16k3(**kwargs):
    #1116426
    return ConvMixer(256, 16, kernel_size=3, **kwargs)


def convmixer256d16k9(**kwargs):

    return ConvMixer(256, 16, kernel_size=9, **kwargs)


def convmixer256d8(**kwargs):
    return ConvMixer(256, 8, **kwargs)


def convmixer256d16(**kwargs):
    return ConvMixer(256, 16, **kwargs)


def convmixer1024d10(**kwargs):
    # 11540490
    return ConvMixer(1024, 10, **kwargs)


def convmixer768d17(**kwargs):
    # 11285770
    return ConvMixer(768, 17, **kwargs)


def convmixer1024d18(**kwargs):
    return ConvMixer(1024, 18, **kwargs)


def convmixer2048d5(**kwargs):
    return ConvMixer(2048, 5, **kwargs)
