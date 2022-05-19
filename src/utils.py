import torch
import wandb
from torchvision.utils import make_grid


def tensor2image(x):
    return (0.5 * (x + 1)).clip(0, 1)


def wandb_save_images(images):
    grid = make_grid(images, nrow=1, normalize=True, padding=2)
    grid = wandb.Image(grid, caption="src | ref | transfer")
    wandb.log({"transfer": grid})
