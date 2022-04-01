from copy import deepcopy

import torch
import torch.nn as nn

from src.utils import tensor2image


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
    def __init__(self):
        super(HistogramLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, input_data, target_data, mask_src, mask_tar, index, ref):
        """
        better calc indices here (see dataset.py mask_preprocess method)
        as these indices do not allow for batch constructions
        either remove from dataset or pad with -1 to unisize

        anyway this loss will be computed with for-loop over batch
        because histogram matching does not work for batch
        """

        input_data = (tensor2image(input_data) * 255).squeeze()
        target_data = (tensor2image(target_data) * 255).squeeze()
        ref = (tensor2image(ref) * 255).squeeze()

        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()

        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        ref_masked = ref * mask_src

        with torch.no_grad():
            input_match = histogram_matching(ref_masked, target_masked, index).to(input_masked.device) * mask_src

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


def histogram_matching(dst_img, ref_img, index):
    """
    perform histogram matching
    dstImg is transformed to have the same the histogram with refImg's
    index[0], index[1]: the index of pixels that need to be transformed in dstImg
    index[2], index[3]: the index of pixels that to compute histogram in refImg
    """

    # faster but less accurate approach
    # dst_img = dst_img.cpu().numpy()
    # ref_img = ref_img.cpu().numpy()
    # matched = match_histograms(dst_img, ref_img, channel_axis=0)
    # return torch.from_numpy(matched)

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

    return torch.tensor(dst_img, dtype=torch.float32)
