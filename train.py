#!/usr/bin/env python


import os
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from src.SCGAN import SCGAN
from src.dataset import SCDataset


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    with open(args.config, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    cfg["wandb"]["local"] = args.local
    cfg = init_wandb(cfg)

    dataset = SCDataset(**cfg["img_info"], **cfg["dataset"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=args.num_workers)

    model = SCGAN(phase=cfg["phase"], **cfg["img_info"], **cfg["model"]).to(args.device)
    model.fit(dataloader, **cfg["fit"])


if __name__ == "__main__":
    main()
