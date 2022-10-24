import models


def get_lowcase_callable_dict(model_dict):
    return {
        name: v
        for name, v in model_dict.__dict__.items() if name.islower()
                                                      and not name.startswith("__") and callable(
            model_dict.__dict__[name])
    }


def print_numel(net):
    net_ = net()
    if net_ is not None:
        print(net.__name__, sum(p.numel() for p in net_.parameters()) / 1000000)


if __name__ == '__main__':
    for net in get_lowcase_callable_dict(models).values():
        print_numel(net)
