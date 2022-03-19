import os
from datetime import datetime

import wandb


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
