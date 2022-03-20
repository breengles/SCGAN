import os
from datetime import datetime

import torch
import wandb
from torchvision.utils import make_grid

from src.utils import tensor2image


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


def save_images(images):
    imgs = tensor2image(torch.cat(images, dim=0))
    grid = make_grid(imgs, nrow=imgs.shape[0] // 3, normalize=True)
    grid = wandb.Image(grid, caption="top: src, mid: ref, bot: transfer")
    wandb.log({"transfer": grid})
