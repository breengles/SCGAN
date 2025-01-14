import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, hsv=False, fast_matching=False, l1=True):
        super().__init__()

        self.hsv = hsv
        self.fast_matching = fast_matching

        if l1:
            self.criterion = torch.nn.L1Loss()
        else:
            self.criterion = torch.nn.MSELoss()

    def forward(self, input_data, target_data, mask_src, mask_tar, ref_data):
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
            target = rgb2hsv(target)
            ref = rgb2hsv(ref)

        mask_src = mask_src.expand(mask_src.size(0), 3, mask_src.size(2), mask_src.size(2))
        mask_tar = mask_tar.expand(mask_src.size(0), 3, mask_tar.size(2), mask_tar.size(2))

        input_masked = input * mask_src
        target_masked = target * mask_tar
        ref_masked = ref * mask_src

        input_match = self.match_histogram(ref_masked, target_masked).to(input_masked.device) * mask_src

        if self.hsv:
            input_match = hsv2rgb(input_match)

        return self.criterion(input_masked, input_match)

    @torch.no_grad()
    def match_histogram(self, source, reference):
        """
        Adjust the pixel values of images such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            reference: np.ndarray
                Reference image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """
        source = source.cpu().numpy()
        reference = reference.cpu().numpy()

        oldshape = source.shape
        batch_size = oldshape[0]
        # get the set of unique pixel values and their corresponding indices and
        # counts
        result = np.zeros_like(source, dtype=np.uint8)
        for i in range(batch_size):
            for c in range(3):
                s = source[i, c].ravel()
                r = reference[i, c].ravel()

                s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
                r_values, r_counts = np.unique(r, return_counts=True)

                if len(s_counts) == 1 or len(r_counts) == 1:
                    continue
                # take the cumsum of the counts and normalize by the number of pixels to
                # get the empirical cumulative distribution functions for the source and
                # template images (maps pixel value --> quantile)
                s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
                s_quantiles /= s_quantiles[-1]
                r_quantiles = np.cumsum(r_counts[1:]).astype(np.float64)
                r_quantiles /= r_quantiles[-1]
                r_values = r_values[1:]

                # interpolate linearly to find the pixel values in the template image
                # that correspond most closely to the quantiles in the source image
                interp_value = np.zeros_like(s_values, dtype=np.float32)
                interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
                interp_value[1:] = interp_r_values
                result[i, c] = interp_value[bin_idx].reshape(oldshape[2], oldshape[3])

        return torch.from_numpy(result).to(torch.float32)
