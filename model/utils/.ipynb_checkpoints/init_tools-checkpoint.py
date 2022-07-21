import torch.nn as nn


def init_layer(layer, mean, std, truncated=False):
    if truncated:
        nn.init.normal_(layer.weight.data, mean=mean, std=std)
        pass
    else:
        nn.init.normal_(layer.weight.data, mean=mean, std=std)
        nn.init.zeros_(layer.bias.data)
        pass