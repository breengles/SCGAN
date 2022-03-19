from torch import nn

from src.modules import FaceEncoder, MLP, MakeupFuseDecoder, PartStyleEncoder
from src.utils import Activation, Norm, assign_adain_params, get_num_adain_params


class SCGen(nn.Module):
    def __init__(
        self,
        dim,
        style_dim,
        num_downsamplings,
        num_residuals,
        mlp_dim,
        n_components,
        inp_dim,
        activation=Activation.RELU,
        padding_mode="reflect",
    ):
        super().__init__()

        self.PSEnc = PartStyleEncoder(
            inp_dim=inp_dim,
            inner_dim=dim,
            style_dim=int(style_dim / n_components),
            norm=Norm.NONE,
            activation=activation,
            padding_mode=padding_mode,
        )

        self.FIEnc = FaceEncoder(
            num_downsamplings=num_downsamplings,
            num_residuals=num_residuals,
            inp_dim=inp_dim,
            inner_dim=dim,
            norm=Norm.IN,
            activation=activation,
            padding_mode=padding_mode,
        )

        self.MFDec = MakeupFuseDecoder(
            num_upsamplings=num_downsamplings,
            num_residuals=num_residuals,
            inner_dim=self.FIEnc.out_dim,
            out_dim=inp_dim,
            residual_norm=Norm.ADAIN,
            activation=activation,
            padding_mode=padding_mode,
        )

        self.MLP = MLP(
            inp_dim=style_dim,
            inner_dim=mlp_dim,
            out_dim=get_num_adain_params(self.MFDec),
            num_blocks=3,
            norm=Norm.NONE,
            activation=activation,
        )

    def forward(self, x, y1, map_y1):
        fid_x = self.FIEnc(x)
        code = self.PSEnc(y1, map_y1)
        return self.fuse(fid_x, code, code)

    def fuse(self, content, style_1, style_2, alpha=0):
        adain_params = self.MLP(style_1, style_2, alpha)

        assign_adain_params(self.MFDec, adain_params)
        return self.MFDec(content)
