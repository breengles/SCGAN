import torch
import wandb
from torchvision.utils import make_grid


def tensor2image(x):
    return (0.5 * (x + 1)).clip(0, 1)


def wandb_save_images(images):
    imgs = tensor2image(torch.cat(images, dim=0))
    grid = make_grid(imgs, nrow=imgs.shape[0] // 3, normalize=True)
    grid = wandb.Image(grid, caption="top: src, mid: ref, bot: transfer")
    wandb.log({"transfer": grid})
