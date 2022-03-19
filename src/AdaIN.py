import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = None
        self.bias = None

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"

        batch_size, num_channel, *_ = x.shape
        running_mean = self.running_mean.repeat(batch_size)
        running_var = self.running_var.repeat(batch_size)

        # Apply instance norm
        x_reshaped = x.reshape(1, batch_size * num_channel, *x.shape[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)

        return out.reshape(batch_size, num_channel, *x.shape[2:])
