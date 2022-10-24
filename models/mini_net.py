import torch
from torch import nn


class LocalAttention(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniNet, self).__init__()
        self.conv1 = self.make_conv_norm_activate(3, 32, kernel_size=3, stride=1, padding=1,
                                                  bias=False)
        self.conv2 = self.make_conv_norm_activate(32, 32, kernel_size=3, stride=1, padding=1,
                                                  bias=False)
        self.conv3 = self.make_conv_norm_activate(32, 32, kernel_size=3, stride=1, padding=1,
                                                  bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out) + out
        out = self.conv3(out) + out
        out = self.avg_pool(out)
        out = self.flatten(out)
        return self.fc(out)

    @staticmethod
    def __make_conv_norm_activate(num_features, inplanes, kernel_size=3, stride=1, padding=1,
                                  bias=False, norm_layer=None, activate_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d(inplanes)
        if activate_layer is None:
            activate_layer = nn.ReLU(inplace=True)
        return nn.Sequential(
            nn.Conv2d(num_features, inplanes, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            norm_layer,
            activate_layer
        )


class MiniNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniNet, self).__init__()
        self.conv1 = self.make_conv_norm_activate(3, 32, kernel_size=3, stride=1, padding=1,
                                                  bias=False)
        self.conv2 = self.make_conv_norm_activate(32, 32, kernel_size=3, stride=1, padding=1,
                                                  bias=False)
        self.conv3 = self.make_conv_norm_activate(32, 32, kernel_size=3, stride=1, padding=1,
                                                  bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out) + out
        out = self.conv3(out) + out
        out = self.avg_pool(out)
        out = self.flatten(out)
        return self.fc(out)

    @staticmethod
    def make_conv_norm_activate(num_features, inplanes, kernel_size=3, stride=1, padding=1,
                                bias=False, norm_layer=None, activate_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d(inplanes)
        if activate_layer is None:
            activate_layer = nn.ReLU(inplace=True)
        return nn.Sequential(
            nn.Conv2d(num_features, inplanes, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            norm_layer,
            activate_layer
        )


def mini_net(num_classes=10):
    return MiniNet(num_classes)


if __name__ == '__main__':
    """
    """
    net = mini_net().cuda()

    net(torch.zeros(5, 3, 32, 32).cuda())
