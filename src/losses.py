import torch
import torch.nn as nn
from skimage.exposure import match_histograms

from src.utils import tensor2image


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, device="cpu"):
        super().__init__()

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

        self.device = device

    def forward(self, x, label):
        targets = torch.ones_like(x, device=self.device) * label
        return self.loss(x, targets)


class HistogramLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, src_img, target_img, src_mask, target_mask, ref_img):
        src_img = tensor2image(src_img)
        target_img = tensor2image(target_img)
        ref_img = tensor2image(ref_img)

        src_mask = src_mask.expand(-1, 3, src_mask.shape[2], src_mask.shape[3])
        target_mask = target_mask.expand(-1, 3, target_mask.shape[2], target_mask.shape[3])

        src_masked = src_img * src_mask
        target_masked = target_img * target_mask
        ref_masked = ref_img * src_mask

        matched = []
        for ref, target, mask in zip(ref_masked, target_masked, src_mask):
            matched.append(self.histogram_matching(ref, target) * mask)  # TODO: check correctness
        matched = torch.vstack(matched)

        return self.loss(src_masked, matched)

    @torch.no_grad()
    def histogram_matching(self, src_img, ref_img):
        """
        perform histogram matching
        src_img is transformed to have the same the histogram with ref_img's
        index[0], index[1]: the index of pixels that need to be transformed in src_img
        index[2], index[3]: the index of pixels that to compute histogram in ref_img
        """
        src_align = src_img.cpu().numpy().transpose(1, 2, 0)
        ref_align = ref_img.cpu().numpy().transpose(1, 2, 0)

        matched = match_histograms(src_align, ref_align, multichannel=True).transpose(2, 0, 1)

        return torch.tensor(matched, dtype=torch.float32, device=self.device).unsqueeze(0)
