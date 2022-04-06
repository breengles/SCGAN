#!/usr/bin/env python


from argparse import ArgumentParser

import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.SCGen import SCGen
from src.dataset import TransferDataset
from src.utils import tensor2image


def main():
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--n_threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scgen", type=str, default=None)

    args = parser.parse_args()

    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)

    dataset = TransferDataset(**cfg["img_info"], **cfg["dataset"])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=args.n_threads)

    model = SCGen(
        dim=cfg["model"]["ngf"],
        style_dim=cfg["model"]["style_dim"],
        n_downsample=cfg["model"]["n_downsampling"],
        n_res=cfg["model"]["n_res"],
        mlp_dim=cfg["model"]["mlp_dim"],
        n_componets=cfg["img_info"]["n_components"],
        input_dim=cfg["model"]["input_nc"],
        phase="transfer",
        ispartial=cfg["model"]["ispartial"],
        isinterpolation=cfg["model"]["isinterpolation"],
        vgg_root=cfg["model"]["vgg_root"],
    )

    if args.scgen is not None:
        state_dict = torch.load(args.scgen)
    else:
        state_dict = torch.load(cfg["model"]["pretrained_scgen_path"])

    if "SCGen_state_dict" in state_dict.keys():
        model.load_state_dict(state_dict["SCGen_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.to(args.device).eval()

    for data in tqdm(dataloader):
        makeup = data["makeup_img"].to(args.device)
        nonmakeup = data["nonmakeup_img"].to(args.device)
        makeup_seg = data["makeup_seg"].to(args.device)

        with torch.no_grad():
            result = model.transfer(nonmakeup, makeup, makeup_seg)

        nonmakeup = tensor2image(nonmakeup).squeeze(0).permute(1, 2, 0).cpu().numpy()
        makeup = tensor2image(makeup).squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = tensor2image(result).squeeze(0).permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(nonmakeup)
        ax[1].imshow(makeup)
        ax[2].imshow(result)

        plt.show()

        exit()


if __name__ == "__main__":
    main()
