from copy import deepcopy

import torch
import torch.nn as nn
from datetime import datetime

# from skimage.exposure import match_histogram


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

        self.loss = nn.L1Loss()
        self.device = device

    @staticmethod
    def denorm(x):
        out = (x + 1) * 0.5
        return out.clip(0, 1) * 255

    def forward(self, x, y, mask_src, mask_target, index, ref):
        x = self.denorm(x)
        y = self.denorm(y)
        ref = self.denorm(ref)

        mask_src = mask_src.expand(-1, 3, mask_src.shape[2], mask_src.shape[2])
        mask_target = mask_target.expand(-1, 3, mask_target.shape[2], mask_target.shape[2])

        input_masked = x * mask_src
        target_masked = y * mask_target
        ref_masked = ref * mask_src

        input_match = []
        for ref, target, idx in zip(ref_masked, target_masked, index):
            input_match.append(self.histogram_matching(ref, target, idx))  # TODO: batch support
        input_match = torch.vstack(input_match)

        return self.loss(input_masked, input_match)

    @torch.no_grad()
    def histogram_matching(self, src_img, ref_img, index):
        """
        perform histogram matching
        src_img is transformed to have the same the histogram with ref_img's
        index[0], index[1]: the index of pixels that need to be transformed in src_img
        index[2], index[3]: the index of pixels that to compute histogram in ref_img
        """
        index = [x[x != -1] for x in index]

        src_img = src_img
        ref_img = ref_img

        dst_align = src_img[:, index[0], index[1]]
        ref_align = ref_img[:, index[2], index[3]]

        hist_ref = self.calc_histogram(ref_align)
        hist_dst = self.calc_histogram(dst_align)

        tables = [self.calc_transfer_func(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

        mid = deepcopy(dst_align)

        for i in range(3):
            for k in range(0, len(index[0])):
                dst_align[i][k] = tables[i][int(mid[i][k])]

            src_img[i, index[0], index[1]] = dst_align[i]

        return src_img.unsqueeze(0)

    @staticmethod
    def calc_histogram(image):
        """
        cal cumulative hist for channel list
        """

        hists = []
        for i in range(3):
            channel = image[i]
            hist = torch.histc(channel, bins=256, min=0, max=256)
            sum = hist.sum()
            pdf = [v / sum for v in hist]

            for i in range(1, 256):
                pdf[i] = pdf[i - 1] + pdf[i]

            hists.append(pdf)

        return hists

    @staticmethod
    def calc_transfer_func(ref, adj):
        """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
        """

        table = list(range(0, 256))

        for i in list(range(1, 256)):
            for j in list(range(1, 256)):
                if adj[j - 1] <= ref[i] <= adj[j]:
                    table[i] = j
                    break

        table[255] = 255

        return table
