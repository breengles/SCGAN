from copy import deepcopy

import torch
import torch.nn as nn
from skimage.exposure import match_histograms

from src.utils import tensor2image


def rgb2hsv(input, epsilon=1e-10):
    assert input.shape[1] == 3

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)


def hsv2rgb(input):
    assert input.shape[1] == 3

    h, s, v = input[:, 0], input[:, 1], input[:, 2]
    h_ = (h - torch.floor(h / 360) * 360) / 60
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack(
        (
            torch.stack((c, x, zero), dim=1),
            torch.stack((x, c, zero), dim=1),
            torch.stack((zero, c, x), dim=1),
            torch.stack((zero, x, c), dim=1),
            torch.stack((x, zero, c), dim=1),
            torch.stack((c, zero, x), dim=1),
        ),
        dim=0,
    )

    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
    return rgb


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super().__init__()

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def forward(self, x, target_is_real):
        target_tensor = torch.ones_like(x) * target_is_real
        return self.loss(x, target_tensor)


class HistogramLoss(nn.Module):
    def __init__(self, hsv=False, fast_matching=False):
        super().__init__()

        self.hsv = hsv
        self.fast_matching = fast_matching

        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.MSELoss()

    def forward(self, input_data, target_data, mask_src, mask_tar, index, ref_data):
        """
        better calc indices here (see dataset.py mask_preprocess method)
        as these indices do not allow for batch constructions
        either remove from dataset or pad with -1 to unisize

        anyway this loss will be computed with for-loop over batch
        because histogram matching does not work in that way
        """
        input = input_data.clone()
        target = target_data.clone()
        ref = ref_data.clone()

        input = tensor2image(input) * 255
        target = tensor2image(target) * 255
        ref = tensor2image(ref) * 255

        if self.hsv:
            # input = rgb2hsv(input)
            target = rgb2hsv(target)
            ref = rgb2hsv(ref)

        # index_src = torch.nonzero(mask_src)
        # index_tar = torch.nonzero(mask_tar)
        # index_src_x = index_src[:, 1]
        # index_src_y = index_src[:, 2]
        # index_tar_x = index_tar[:, 1]
        # index_tar_y = index_tar[:, 2]

        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2))
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2))

        input_masked = input * mask_src
        target_masked = (target * mask_tar).squeeze()
        ref_masked = (ref * mask_src).squeeze()

        with torch.no_grad():
            # TODO: for-loop over batch with extracted indices
            input_match = (
                histogram_matching(ref_masked, target_masked, index, fast_matching=self.fast_matching).to(
                    input_masked.device
                )
                * mask_src
            )

        if self.hsv:
            input_match = hsv2rgb(input_match)

        return self.criterion(input_masked, input_match)


def cal_hist(image):
    """
    cal cumulative hist for channel list
    """

    hists = []
    for i in range(3):
        channel = image[i]
        channel = torch.from_numpy(channel)
        hist = torch.histc(channel, bins=256, min=0, max=256).numpy()

        cdf = hist / hist.sum()
        for j in range(1, 256):
            cdf[j] = cdf[j - 1] + cdf[j]

        hists.append(cdf)

    return hists


def cal_trans(ref, adj):
    """
    calculate transfer function
    algorithm refering to wiki item: Histogram matching
    """

    table = list(range(256))

    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if adj[j - 1] <= ref[i] <= adj[j]:
                table[i] = j
                break

    table[255] = 255

    return table


def histogram_matching(dst_img, ref_img, index, fast_matching=False):
    """
    perform histogram matching
    dstImg is transformed to have the same the histogram with refImg's
    index[0], index[1]: the index of pixels that need to be transformed in dstImg
    index[2], index[3]: the index of pixels that to compute histogram in refImg
    """

    # faster but less accurate approach
    if fast_matching:
        dst_img = dst_img.cpu().numpy()
        ref_img = ref_img.cpu().numpy()
        matched = match_histograms(dst_img, ref_img, channel_axis=0)
        return torch.from_numpy(matched).unsqueeze(0)
    else:
        index = [x.cpu().numpy().squeeze(0) for x in index]

        dst_img = dst_img.detach().cpu().numpy()
        ref_img = ref_img.detach().cpu().numpy()

        dst_align = dst_img[:, index[0], index[1]]
        ref_align = ref_img[:, index[2], index[3]]

        hist_ref = cal_hist(ref_align)
        hist_dst = cal_hist(dst_align)

        tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(3)]

        mid = deepcopy(dst_align)

        for i in range(3):
            for k in range(len(index[0])):
                dst_align[i, k] = tables[i][int(mid[i, k])]

        for i in range(3):
            dst_img[i, index[0], index[1]] = dst_align[i]

        return torch.tensor(dst_img, dtype=torch.float32).unsqueeze(0)
