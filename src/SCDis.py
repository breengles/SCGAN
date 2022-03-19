import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
from src.utils import Norm


class SCDis(nn.Module):
    def __init__(self, img_size, conv_dim=64, norm=Norm.SN):
        super().__init__()

        layers = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]

        curr_dim = conv_dim
        repeat_num = int(np.log2(img_size) - 2)
        for _ in range(repeat_num):
            layers.extend(
                [nn.Conv2d(curr_dim, curr_dim + conv_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            )
            curr_dim += conv_dim

        # layers.extend([nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1), nn.LeakyReLU(0.01)])
        # curr_dim *= 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))

        if norm == Norm.SN:
            for idx, l in enumerate(layers):
                if isinstance(l, nn.Conv2d):
                    layers[idx] = spectral_norm(l)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).squeeze()


if __name__ == "__main__":
    import torch

    img_size = 512

    img = torch.randn((4, 3, img_size, img_size)).to("cuda")
    disc = SCDis(img_size, conv_dim=16).to("cuda")
    out = disc(img)
    print(out.shape)
