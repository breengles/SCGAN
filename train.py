#!/usr/bin/env python

from argparse import ArgumentParser

import yaml
from torch.utils.data import DataLoader

from src.SCGAN import SCGAN
from src.dataset import SCDataset, determine_eye_box
from src.log_utils import init_wandb
from src.utils import determine_norm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    cfg["wandb"]["local"] = args.local
    cfg = init_wandb(cfg)

    cfg["model"]["norm"] = determine_norm(cfg["model"]["norm"])
    cfg["dataset"]["eye_box"] = determine_eye_box(cfg["dataset"]["eye_box"])

    img_size = cfg["img_size"]

    dataset = SCDataset(img_size=img_size, **cfg["dataset"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = SCGAN(img_size=img_size, **cfg["model"]).to(args.device)
    model.fit(dataloader, **cfg["fit"])
