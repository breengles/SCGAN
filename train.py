#!/usr/bin/env python

from argparse import ArgumentParser

import yaml
from torch.utils.data import DataLoader

from src.SCGAN import SCGAN
from src.dataset import SCDataset
from src.log_utils import init_wandb


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    with open(args.config, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    cfg = init_wandb(cfg)

    dataroot = cfg["dataroot"]
    img_size = cfg["img_size"]

    dataset = SCDataset(dataroot, img_size)
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = SCGAN(img_size=img_size, **cfg["model"]).to(args.device)
    model.fit(dataloader, epochs=cfg["epochs"])
