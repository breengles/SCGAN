import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class SCDis(nn.Module):
    def __init__(self, img_size=512, conv_dim=64, repeat_num=3, norm="sn"):
        super().__init__()

        layers = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]

        curr_dim = conv_dim
        for _ in range(repeat_num):
            layers.extend(
                [
                    nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.01, inplace=True),
                ]
            )
            curr_dim *= 2

        layers.extend(
            [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1), nn.LeakyReLU(0.01, inplace=True)]
        )
        curr_dim *= 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))

        if norm.lower() == "sn":
            for idx, l in layers:
                if isinstance(l, nn.Conv2d):
                    layers[idx] = spectral_norm(l)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
