'''
Resnet for CIFAR
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 groups=1,
                 wide_rate=1,
                 activation=None,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes,
                               planes * wide_rate,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = norm_layer(planes * wide_rate)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(planes * wide_rate,
                               self.expansion * planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=groups,
                               bias=False)
        self.bn2 = norm_layer(self.expansion * planes)
        self.act2 = activation()

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), norm_layer(self.expansion * planes))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 groups=1,
                 wide_rate=1,
                 activation=None,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes,
                               planes * wide_rate,
                               kernel_size=1,
                               bias=False)
        self.bn1 = norm_layer(planes * wide_rate)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(planes * wide_rate,
                               planes * wide_rate,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=groups,
                               bias=False)
        self.bn2 = norm_layer(planes * wide_rate)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(planes * wide_rate,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(self.expansion * planes)
        self.act3 = activation()

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), norm_layer(self.expansion * planes))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            num_blocks,
            num_classes=10,
            wf=1,  # for widen factor of wide resnet
            wr=1,  # for inner wide rate of resnext
            groups=1,  # for cardinality of resnext
            activation=None,
            norm_layer=None):
        super(ResNet, self).__init__()
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        widths = [int(w * wf) for w in [64, 128, 256, 512]]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3,
                               self.in_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.act1 = activation()
        self.layer1 = self._make_layer(block,
                                       widths[0],
                                       num_blocks[0],
                                       stride=1,
                                       groups=groups,
                                       wide_rate=wr,
                                       activation=activation)
        self.layer2 = self._make_layer(block,
                                       widths[1],
                                       num_blocks[1],
                                       stride=2,
                                       groups=groups,
                                       wide_rate=wr,
                                       activation=activation)
        self.layer3 = self._make_layer(block,
                                       widths[2],
                                       num_blocks[2],
                                       stride=2,
                                       groups=groups,
                                       wide_rate=wr,
                                       activation=activation)
        self.layer4 = self._make_layer(block,
                                       widths[3],
                                       num_blocks[3],
                                       stride=2,
                                       groups=groups,
                                       wide_rate=wr,
                                       activation=activation)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(widths[3] * block.expansion, num_classes)

    def _make_layer(self,
                    block,
                    planes,
                    num_blocks,
                    stride,
                    groups,
                    wide_rate,
                    activation=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes,
                      planes,
                      stride,
                      groups=groups,
                      wide_rate=wide_rate,
                      activation=activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


# simple resnet
def resnet10(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes**kwargs)


def resnet18(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def resnet34(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet50(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet101(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def resnet152(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


#resnext
def resnext10(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1],
                  wr=2,
                  groups=32,
                  num_classes=num_classes**kwargs)


def resnext18(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  wr=2,
                  groups=32,
                  num_classes=num_classes,
                  **kwargs)


def resnext34(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  wr=2,
                  groups=32,
                  num_classes=num_classes,
                  **kwargs)


def resnext50(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  wr=2,
                  groups=32,
                  num_classes=num_classes,
                  **kwargs)


def resnext101(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  wr=2,
                  groups=32,
                  num_classes=num_classes,
                  **kwargs)


def resnext152(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3],
                  wr=2,
                  groups=32,
                  num_classes=num_classes,
                  **kwargs)


# wider resnet
def widerres14(num_classes=10, **kwargs):
    """
    compare to resnet34
    """
    return ResNet(BasicBlock, [2, 2, 1, 1],
                  num_classes=num_classes,
                  wf=2,
                  **kwargs)


#resnet with different activation
def resnet10_celu(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1],
                  num_classes=num_classes,
                  activation=lambda: nn.CELU(inplace=True),
                  **kwargs)


def resnet18_celu(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes,
                  activation=lambda: nn.CELU(inplace=True),
                  **kwargs)


def resnet18_gelu(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes,
                  activation=nn.GELU,
                  **kwargs)


def resnet18_leakyrelu(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes,
                  activation=lambda: nn.LeakyReLU(inplace=True),
                  **kwargs)


#resnet with different norm layers


def resnet18_in(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes,
                  norm_layer=nn.InstanceNorm2d,
                  **kwargs)


def resnet18_ln(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes,
                  norm_layer=lambda x: nn.GroupNorm(1, x),
                  **kwargs)


def resnet18_gn(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes,
                  norm_layer=lambda x: nn.GroupNorm(32, x),
                  **kwargs)


def main():
    net = resnet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == '__main__':
    """
    """
    print(sum(p.numel() for p in resnet18().parameters()))  #11173962
    print(sum(p.numel() for p in resnet34().parameters()))  #21282122
    print(sum(p.numel() for p in widerres14().parameters()))  #21063818
