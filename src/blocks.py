from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from src.utils import Activation, Norm, get_activation, get_norm


class FCBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, bias=True, norm=Norm.NONE, activation=Activation.RELU):
        super().__init__()

        self.fc = nn.Linear(inp_dim, out_dim, bias=bias)

        if norm == Norm.SN:
            self.fc = spectral_norm(self.fc)

        self.norm = get_norm(norm, out_dim)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.fc(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        padding_mode="zeros",
        bias=True,
        norm=Norm.NONE,
        activation=Activation.NONE,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        )

        if norm == Norm.SN:
            self.conv = spectral_norm(self.conv)

        self.norm = get_norm(norm, out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm=Norm.IN, activation=Activation.RELU, padding_mode="zeros", bias=True):
        super().__init__()

        self.main = nn.Sequential(
            ConvBlock(dim, dim, 3, 1, 1, padding_mode, bias, norm=norm, activation=activation),
            ConvBlock(dim, dim, 3, 1, 1, padding_mode, bias),
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm=Norm.IN, activation=Activation.RELU, padding_mode="zeros", bias=True):
        super().__init__()

        blocks = [ResidualBlock(dim, norm, activation, padding_mode, bias) for _ in range(num_blocks)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        return self.main(x)
