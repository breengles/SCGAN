from enum import Enum, auto

import torch
from torch import nn
from torch.nn import init

from src.AdaIN import AdaptiveInstanceNorm2d


class Norm(Enum):
    SN = auto()
    BN = auto()
    IN = auto()
    LN = auto()
    NONE = auto()
    ADAIN = auto()


class Activation(Enum):
    RELU = auto()
    LRELU = auto()
    PRELU = auto()
    SELU = auto()
    TANH = auto()
    NONE = auto()


def get_norm(norm, dim):
    if norm in (Norm.NONE, Norm.SN):
        return None
    elif norm == Norm.BN:
        return nn.BatchNorm2d(dim)
    elif norm == Norm.IN:
        return nn.InstanceNorm2d(dim)
    elif norm == Norm.LN:
        # return nn.LayerNorm(dim)
        return LayerNorm(dim)
    elif norm == Norm.ADAIN:
        return AdaptiveInstanceNorm2d(dim)
    else:
        raise ValueError(f"Unsupported norm type {norm}")


def get_activation(activation):
    if activation == Activation.RELU:
        return nn.ReLU()
    elif activation == Activation.LRELU:
        return nn.LeakyReLU(0.2)
    elif activation == Activation.PRELU:
        return nn.PReLU()
    elif activation == Activation.SELU:
        return nn.SELU()
    elif activation == Activation.TANH:
        return nn.Tanh()
    elif activation == Activation.NONE:
        return None
    else:
        raise ValueError(f"Unsupported activation type {activation}")


def get_num_adain_params(model):
    out = 0
    for module in model.modules():
        if module.__class__.__name__ == "AdaptiveInstanceNorm2d":
            out += 2 * module.num_features
    return out


def assign_adain_params(model, adain_params):
    for module in model.modules():
        if module.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_features = module.num_features

            mean = adain_params[:, :num_features]
            std = adain_params[:, num_features : 2 * num_features]

            module.bias = mean.reshape(-1).contiguous()
            module.weight = std.reshape(-1).contiguous()

            if adain_params.shape[1] > 2 * num_features:
                adain_params = adain_params[:, 2 * num_features :]


def xavier_init(model):
    classname = model.__class__.__name__

    if classname.find("Conv2d") != -1:
        init.xavier_normal_(model.weight.data, gain=0.02)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(model.weight.data, gain=0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal(model.weight.data, 1.0, 0.02)
        init.constant(model.bias.data, 0.0)


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)

        mean = x.flatten(1).mean(1).reshape(*shape)
        std = x.flatten(1).std(1).reshape(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.reshape(*shape) + self.beta.reshape(*shape)

        return x
