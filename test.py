#!/usr/bin/env python

import yaml
from torch.utils.data import DataLoader

from src.dataset import SCDataset
from src.log_utils import init_wandb
from src.SCGAN import SCGAN

if __name__ == "__main__":
    with open("configs/test.yaml", "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    cfg = init_wandb(cfg)

    dataroot = "datasets/daniil.128px.cut.crop.overfit"
    img_size = 128

    dataset = SCDataset(cfg["dataroot"], cfg["img_size"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = SCGAN(img_size=cfg["img_size"]).to(cfg["device"])
    model.fit(dataloader)
