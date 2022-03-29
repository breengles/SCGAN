#!/usr/bin/env python


import os
from datetime import datetime

import wandb
from src.models import create_model
from src.options.test_options import TestOptions
from src.dataset import SCDataLoader


def init_wandb(opt):
    if opt.wb_local:
        os.environ["WANDB_MODE"] = "offline"
    else:
        wandb.login()

    now = datetime.now()
    wandb.init(
        project=opt.wb_project,
        name=f"{opt.wb_name}:{now.hour}:{now.minute}:{now.second}-{now.day}.{now.month}.{now.year}",
        group=opt.wb_group,
        notes=opt.wb_notes,
        entity=opt.wb_entitiy,
    )
    return wandb.config


def main():
    opt = TestOptions().parse()
    data_loader = SCDataLoader(opt)
    SCGan = create_model(opt, data_loader)

    wb_cfg = init_wandb(opt)

    if opt.phase == "train":
        SCGan.train()
    elif opt.phase == "test":
        SCGan.test()

    print("Finished!!!")


if __name__ == "__main__":
    main()
