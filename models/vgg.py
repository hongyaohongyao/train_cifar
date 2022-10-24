"""
VGG11/13/16/19 in Pytorch.
"""

import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'VGG19': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


class VGG(nn.Module):

    def __init__(self,
                 vgg_name,
                 num_classes=10,
                 activation=None,
                 norm_layer=None):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, with_latent=False):
        out = self.features(x)
        latent = self.flatten(out)
        out = self.classifier(latent)
        if with_latent:
            return out, latent
        return out

    def _make_layers(self, cfg, activation=None, norm_layer=None):
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    norm_layer(x),
                    activation()
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(**kwargs):
    return VGG('VGG11', **kwargs)


def vgg13(**kwargs):
    return VGG('VGG13', **kwargs)


def vgg16(**kwargs):
    return VGG('VGG16', **kwargs)


def vgg19(**kwargs):
    return VGG('VGG19', **kwargs)
