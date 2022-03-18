from torch import nn

from blocks import ConvBlock, ResidualBlocks
from utils import Activation, Norm


class MakeupFuseDecoder(nn.Module):
    def __init__(
        self,
        num_upsamplings,
        num_residuals,
        inner_dim,
        out_dim,
        residual_norm=Norm.ADAIN,
        activation=Activation.RELU,
        padding_mode="zero",
    ):
        super().__init__()

        modules = [
            ResidualBlocks(
                num_blocks=num_residuals,
                dim=inner_dim,
                norm=residual_norm,
                activation=activation,
                padding_mode=padding_mode,
            )
        ]

        dim = inner_dim
        for _ in range(num_upsamplings):
            modules.extend(
                [
                    nn.Upsample(scale_factor=2),
                    ConvBlock(dim, dim // 2, 5, 1, 2, norm=Norm.LN, activation=activation, padding_mode=padding_mode,),
                ]
            )
            dim //= 2

        modules.append(
            ConvBlock(dim, out_dim, 7, 1, 3, norm=Norm.NONE, activation=Activation.TANH, padding_mode="reflection")
        )
        self.main = nn.Sequential(*modules)

    def forward(self, x):
        return self.main(x)


class FaceEncoder(nn.Module):
    def __init__(self, num_downsamplings, num_residuals, inp_dim, inner_dim, norm, activation, padding_mode):
        super().__init__()

        modules = [ConvBlock(inp_dim, inner_dim, 7, 1, 3, norm=norm, activation=activation, padding_mode=padding_mode)]

        dim = inner_dim
        for _ in range(num_downsamplings):
            modules.append(
                ConvBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)
            )
            dim *= 2

        modules.append(
            (ResidualBlocks(num_residuals, dim, norm=norm, activation=activation, padding_mode=padding_mode))
        )
        self.main = nn.Sequential(*modules)
        self.out_dim = dim

    def forward(self, x):
        return self.main(x)
