import torch.nn as nn


class RecursiveResidual(nn.Module):

    def __init__(self, fn, depth=11):
        super(RecursiveResidual, self).__init__()
        if depth <= 1:
            self.res = nn.Identity()
        else:
            self.res = RecursiveResidual(fn, depth - 1)

        self.fn = fn()

    def forward(self, x):
        return self.res(x) + self.fn(x)


def RecRes(depth=11,
           dim=1024,
           num_classes=10,
           groups=8,
           norm_layer=None,
           activation=None,
           **kwargs):
    if activation is None:
        activation = lambda: nn.LeakyReLU(inplace=True)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=3, padding=2), norm_layer(dim),
        RecursiveResidual(lambda: nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=groups), activation(),
            norm_layer(dim)),
                          depth=depth), nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(dim, num_classes))


def recres11(**kwargs):
    return RecRes(9, 384, groups=1, **kwargs)
