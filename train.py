#!/usr/bin/env python


import os
from argparse import ArgumentParser
from datetime import datetime

import wandb
import yaml
from torch.utils.data import DataLoader

from src.SCGAN import SCGAN
from src.dataset import SCDataset


def init_wandb(global_cfg):
    cfg = global_cfg["wandb"]

    if cfg["local"]:
        os.environ["WANDB_MODE"] = "offline"
    else:
        wandb.login()

    now = datetime.now()
    wandb.init(
        project=cfg["project"],
        name=f'{cfg["name"]}:{now.hour}:{now.minute}:{now.second}-{now.day}.{now.month}.{now.year}',
        group=cfg.get("group", None),
        notes=cfg["notes"],
        entity=cfg["entity"],
        config=global_cfg,
    )
    return wandb.config


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    cfg["wandb"]["local"] = args.local
    cfg = init_wandb(cfg)

    dataset = SCDataset(**cfg["img_info"], **cfg["dataset"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = SCGAN(phase=cfg["phase"], **cfg["img_info"], **cfg["model"]).to(args.device)
    model.fit(dataloader, **cfg["fit"])


if __name__ == "__main__":
    main()
