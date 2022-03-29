from copy import deepcopy
from matplotlib.pyplot import hist
import torch
import torch.nn as nn
from skimage.exposure import match_histograms
from skimage.exposure.histogram_matching import _match_cumulative_cdf

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
    """
    Fast histogram matching but for whale masked images
    might lead to slightly incorrect matching
    """

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


# class HistogramLoss(nn.Module):
#     """
#     Kinda slow matching for rectangle crops of images
#     """
#     def __init__(self, device="cpu"):
#         super().__init__()

#         self.loss = nn.L1Loss()
#         self.device = device

#     def forward(self, src_img, target_img, src_mask, target_mask, ref_img):
#         src_img = tensor2image(src_img)
#         target_img = tensor2image(target_img)
#         ref_img = tensor2image(ref_img)

#         src_mask = src_mask.expand(-1, 3, src_mask.shape[2], src_mask.shape[3])
#         target_mask = target_mask.expand(-1, 3, target_mask.shape[2], target_mask.shape[3])

#         src_masked = src_img * src_mask
#         target_masked = target_img * target_mask
#         ref_masked = ref_img * src_mask

#         loss = 0.0
#         for ref, target, src, src_m, target_m in zip(ref_masked, target_masked, src_masked, src_mask, target_mask):
#             src_idx = torch.nonzero(src_m, as_tuple=False)
#             target_idx = torch.nonzero(target_m, as_tuple=False)

#             src_idx_x = src_idx[:, 1]
#             src_idx_y = src_idx[:, 2]
#             target_idx_x = target_idx[:, 1]
#             target_idx_y = target_idx[:, 2]

#             ref = ref[:, min(src_idx_x) : max(src_idx_x), min(src_idx_y) : max(src_idx_y)]
#             target = target[:, min(target_idx_x) : max(target_idx_x), min(target_idx_y) : max(target_idx_y)]
#             src = src[:, min(src_idx_x) : max(src_idx_x), min(src_idx_y) : max(src_idx_y)]

#             matched = self.histogram_matching(ref, target)
#             loss += self.loss(src, matched)

#         return loss

#     @torch.no_grad()
#     def histogram_matching(self, src_img, ref_img):
#         """
#         perform histogram matching
#         src_img is transformed to have the same the histogram with ref_img's
#         index[0], index[1]: the index of pixels that need to be transformed in src_img
#         index[2], index[3]: the index of pixels that to compute histogram in ref_img
#         """
#         src_align = src_img.cpu().numpy().transpose(1, 2, 0)
#         ref_align = ref_img.cpu().numpy().transpose(1, 2, 0)

#         matched = match_histograms(src_align, ref_align, multichannel=True).transpose(2, 0, 1)

#         return torch.tensor(matched, dtype=torch.float32, device=self.device)


# class HistogramLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.loss = nn.L1Loss()

#     def forward(self, src, target, src_mask, target_mask, ref):
#         src = self.tensor2image(src)
#         target = self.tensor2image(target)
#         ref = self.tensor2image(ref)

#         print(src.shape, target.shape, ref.shape)

#         src_mask = src_mask.expand(-1, 3, src_mask.shape[2], src_mask.shape[3])
#         target_mask = target_mask.expand(-1, 3, target_mask.shape[2], target_mask.shape[3])

#         src_masked = src * src_mask
#         target_masked = target * target_mask
#         ref_masked = ref * src_mask

#         loss = 0
#         for s, t, r in zip(src_masked, target_masked, ref_masked):
#             s_matched = self.histogram_matching(r, t)
#             loss = self.loss(s, s_matched)

#         return loss

#     def histogram_matching(self, src, target):
#         """
#         perform histogram matching
#         dstImg is transformed to have the same the histogram with refImg's
#         index[0], index[1]: the index of pixels that need to be transformed in dstImg
#         index[2], index[3]: the index of pixels that to compute histogram in refImg
#         """
#         src = src.detach()
#         target = target.detach()
#         out = src.clone()

#         src_index = torch.nonzero(src, as_tuple=False)
#         target_index = torch.nonzero(target, as_tuple=False)

#         src_index_x = src_index[:, 1]
#         src_index_y = src_index[:, 2]
#         target_index_x = target_index[:, 1]
#         target_index_y = target_index[:, 2]

#         src_pixels = src[:, src_index_x, src_index_y]
#         target_pixels = target[:, target_index_x, target_index_y]

#         hist_src = self.cal_hist(src_pixels)
#         hist_target = self.cal_hist(target_pixels)

#         matched_hist = [self.match_cdf(hist_src[i], hist_target[i]) for i in range(3)]

#         # todo

#         return out

#     def match_cdf(self, src, target):
#         out = _match_cumulative_cdf(src, target)
#         return torch.tensor(out)

#     @staticmethod
#     def cal_hist(image):
#         """
#         cal cumulative hist for channel list
#         """

#         hists = []
#         for i in range(3):
#             channel = image[i]
#             hist = torch.histc(channel, bins=256, min=0, max=256)
#             hist = hist / hist.sum()
#             hists.append(hist.cpu().numpy())

#         return hists
