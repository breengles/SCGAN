from enum import Enum, auto

from torch import nn
from AdaIN import AdaptiveInstanceNorm2d


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
        return nn.LayerNorm(dim)
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
