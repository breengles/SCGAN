import torch
from torch import nn
from torchvision import models

from src.blocks import ConvBlock, linearBlock, ResBlocks
from src.utils import Activation, Norm


class MLP(nn.Module):
    def __init__(self, inp_dim, inner_dim, out_dim, num_blocks, norm=Norm.NONE, activation=Activation.RELU):
        super().__init__()

        modules = [
            linearBlock(inp_dim, inp_dim, norm=norm, activation=activation),
            linearBlock(inp_dim, inner_dim, norm=norm, activation=activation),
        ]

        modules.extend(
            [linearBlock(inner_dim, inner_dim, norm=norm, activation=activation) for _ in range(num_blocks - 2)]
        )

        modules.append(linearBlock(inner_dim, out_dim, norm=Norm.NONE, activation=Activation.NONE))

        self.model = nn.Sequential(*modules)

    def forward(self, style_1, style_2, alpha=0):
        x_1 = self.model[:-1](style_1.flatten(1))
        x_2 = self.model[:-1](style_2.flatten(1))

        return self.model[-1]((1 - alpha) * x_1 + alpha * x_2)


class MakeupFuseDecoder(nn.Module):
    def __init__(
        self,
        num_upsamplings,
        num_residuals,
        inner_dim,
        out_dim,
        residual_norm=Norm.ADAIN,
        activation=Activation.RELU,
        padding_mode="zeros",
    ):
        super().__init__()

        modules = [
            ResBlocks(
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
            ConvBlock(dim, out_dim, 7, 1, 3, norm=Norm.NONE, activation=Activation.TANH, padding_mode="reflect")
        )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class FaceEncoder(nn.Module):
    def __init__(self, num_downsamplings, num_residuals, inp_dim, inner_dim, norm, activation, padding_mode):
        super().__init__()

        modules = [ConvBlock(inp_dim, inner_dim, 7, 1, 3, norm=norm, activation=activation, padding_mode=padding_mode)]

        dim = inner_dim
        for _ in range(num_downsamplings):
            modules.append(
                ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)
            )
            dim *= 2

        modules.append((ResBlocks(num_residuals, dim, norm=norm, activation=activation, padding_mode=padding_mode)))
        self.model = nn.Sequential(*modules)
        self.out_dim = dim

    def forward(self, x):
        return self.model(x)


class PartStyleEncoder(nn.Module):
    def __init__(self, inp_dim, inner_dim, style_dim, norm, activation, padding_mode):
        super().__init__()

        self.vgg = None

        dim = inner_dim
        self.conv1 = ConvBlock(inp_dim, dim, 7, 1, 3, norm=norm, activation=activation, padding_mode=padding_mode)
        dim *= 2
        self.conv2 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)
        dim *= 2
        self.conv3 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)
        dim *= 2
        self.conv4 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)
        dim *= 2

        self.model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0))

        # convs = [ConvBlock(inp_dim, inner_dim, 7, 1, 3, norm=norm, activation=activation, padding_mode=padding_mode)]
        # dim = inner_dim * 2
        # for _ in range(3):
        #     convs.append(ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode))
        #     dim *= 2
        #
        # self.convs = nn.ModuleList(convs)
        # self.main = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0))

        self.out_dim = dim

    def load_vgg(self, path):
        vgg19 = models.vgg19(pretrained=True)
        vgg19.load_state_dict(torch.load(path))
        self.vgg = vgg19.features
        self.disable_vgg_grad()

    def disable_vgg_grad(self):
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    @staticmethod
    def get_features(x, model, layers=None):
        if layers is None:
            layers = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1"}

        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    def component_enc(self, x):
        vgg_aux = self.get_features(x, self.vgg)

        x = self.conv1(x)
        x = torch.cat([x, vgg_aux["conv1_1"]], dim=1)
        x = self.conv2(x)
        x = torch.cat([x, vgg_aux["conv2_1"]], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, vgg_aux["conv3_1"]], dim=1)
        x = self.conv4(x)
        x = torch.cat([x, vgg_aux["conv4_1"]], dim=1)

        return self.model(x)

    def forward(self, x, map_x):
        y0 = map_x[:, 0, :, :]
        y0 = y0.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        y0 = x.mul(y0)
        code = self.component_enc(y0)

        for i in range(1, map_x.shape[1]):
            yi = map_x[:, i, :, :]
            yi = yi.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
            yi = x.mul(yi)
            code = torch.cat([code, self.component_enc(yi)], dim=1)

        return code

    @torch.no_grad()
    def global_transfer(self, x, map_x, y1, map_y1, y2, map_y2):
        self.eval()
        raise NotImplementedError()

    @torch.no_grad()
    def partial_transfer(self, x, map_x, y1, map_y1, y2, map_y2):
        self.eval()
        raise NotImplementedError()
