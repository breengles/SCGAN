import os.path
from enum import IntEnum

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import hflip


class Regions(IntEnum):
    FACE = 4
    NOSE = 8
    RIGHT_EYE = 1
    LEFT_EYE = 6
    UPPER_LIP_VERMILLION = 9
    LOWER_LIP_VERMILLION = 13
    RIGHT_EYEBROW = 2
    LEFT_EYEBROW = 7
    TEETH = 11
    NECK = 10
    BACKGROUND = 0
    HAIR = 12


def ToTensor(pic):
    # handle PIL Image
    if pic.mode == "I":
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == "I;16":
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == "YCbCr":
        nchannel = 3
    elif pic.mode == "I;16":
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.permute(2, 0, 1).contiguous()

    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class SCDataset(Dataset):
    def __init__(self, dataroot, img_size, n_components, eye_box="rectangle"):
        self.dataroot = dataroot
        self.img_size = img_size
        self.n_components = n_components
        self.eye_box = eye_box

        self.dir_img = os.path.join(self.dataroot, "images")
        self.dir_img_makeup = os.path.join(self.dir_img, "makeup")
        self.dir_img_nonmakeup = os.path.join(self.dir_img, "non-makeup")

        self.dir_seg = os.path.join(self.dataroot, "segments")
        self.dir_seg_makeup = os.path.join(self.dir_seg, "makeup")
        self.dir_seg_nonmakeup = os.path.join(self.dir_seg, "non-makeup")

        self.makeup_names = os.listdir(self.dir_img_makeup)
        self.non_makeup_names = os.listdir(self.dir_img_nonmakeup)

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.transform_mask = transforms.Compose(
            [transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST), ToTensor]
        )

        self.flip_proba = 0.5

    def __len__(self):
        return len(self.non_makeup_names)

    def pick(self, index):
        nonmakeup_name = self.non_makeup_names[index]

        makeup_index = np.random.randint(0, len(self.makeup_names))
        makeup_name = self.makeup_names[makeup_index]

        return nonmakeup_name, makeup_name

    def __getitem__(self, index):
        nonmakeup_name, makeup_name = self.pick(index)

        makeup_path = os.path.join(self.dir_img_makeup, makeup_name)
        nonmakeup_path = os.path.join(self.dir_img_nonmakeup, nonmakeup_name)

        makeup_img = Image.open(makeup_path).convert("RGB")
        nonmakeup_img = Image.open(nonmakeup_path).convert("RGB")

        makeup_seg_img = Image.open(os.path.join(self.dir_seg_makeup, makeup_name))
        nonmakeup_seg_img = Image.open(os.path.join(self.dir_seg_nonmakeup, nonmakeup_name))

        makeup_img = self.transform(makeup_img)
        nonmakeup_img = self.transform(nonmakeup_img)

        mask_B = self.transform_mask(makeup_seg_img)  # makeup
        mask_A = self.transform_mask(nonmakeup_seg_img)  # nonmakeup

        # augmentations with horizontal flip of image and its mask
        if np.random.uniform() < self.flip_proba:
            makeup_img = hflip(makeup_img)
            mask_B = hflip(mask_B)

        if np.random.uniform() < self.flip_proba:
            nonmakeup_img = hflip(nonmakeup_img)
            mask_A = hflip(mask_A)

        makeup_seg = torch.zeros([self.n_components, self.img_size, self.img_size], dtype=torch.float)
        nonmakeup_seg = torch.zeros([self.n_components, self.img_size, self.img_size], dtype=torch.float)

        makeup_unchanged = (
            (mask_B == Regions.LEFT_EYEBROW).float()
            + (mask_B == Regions.RIGHT_EYEBROW).float()
            + (mask_B == Regions.LEFT_EYE).float()
            + (mask_B == Regions.RIGHT_EYE).float()
            + (mask_B == Regions.TEETH).float()
        )
        nonmakeup_unchanged = (
            (mask_A == Regions.LEFT_EYEBROW).float()
            + (mask_A == Regions.RIGHT_EYEBROW).float()
            + (mask_A == Regions.LEFT_EYE).float()
            + (mask_A == Regions.RIGHT_EYE).float()
            + (mask_A == Regions.TEETH).float()
        )

        mask_A_lip = (mask_A == Regions.UPPER_LIP_VERMILLION).float() + (mask_A == Regions.LOWER_LIP_VERMILLION).float()
        mask_B_lip = (mask_B == Regions.UPPER_LIP_VERMILLION).float() + (mask_B == Regions.LOWER_LIP_VERMILLION).float()
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
        makeup_seg[0] = mask_B_lip
        nonmakeup_seg[0] = mask_A_lip

        mask_A_skin = (
            (mask_A == Regions.FACE).float() + (mask_A == Regions.NOSE).float() + (mask_A == Regions.NECK).float()
        )
        mask_B_skin = (
            (mask_B == Regions.FACE).float() + (mask_B == Regions.NOSE).float() + (mask_B == Regions.NECK).float()
        )
        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
        makeup_seg[1] = mask_B_skin
        nonmakeup_seg[1] = mask_A_skin

        mask_A_eye_left = (mask_A == Regions.LEFT_EYE).float()
        mask_A_eye_right = (mask_A == Regions.RIGHT_EYE).float()
        mask_B_eye_left = (mask_B == Regions.LEFT_EYE).float()
        mask_B_eye_right = (mask_B == Regions.RIGHT_EYE).float()
        mask_A_face = (mask_A == Regions.FACE).float() + (mask_A == Regions.NOSE).float()
        mask_B_face = (mask_B == Regions.FACE).float() + (mask_B == Regions.NOSE).float()

        # avoid the es of ref are closed
        if not ((mask_B_eye_left > 0).any() and (mask_B_eye_right > 0).any()):
            return {}

        # mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(
            mask_A_eye_left, mask_B_eye_left
        )
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(
            mask_A_eye_right, mask_B_eye_right
        )
        makeup_seg[2] = mask_B_eye_left + mask_B_eye_right
        nonmakeup_seg[2] = mask_A_eye_left + mask_A_eye_right

        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["index_A_eye_left"] = index_A_eye_left  # проблемы с индексами, если хотим батч > 1
        mask_A["index_A_eye_right"] = index_A_eye_right
        mask_A["mask_A_skin"] = mask_A_skin
        mask_A["index_A_skin"] = index_A_skin
        mask_A["mask_A_lip"] = mask_A_lip
        mask_A["index_A_lip"] = index_A_lip

        mask_B = {}
        mask_B["mask_B_eye_left"] = mask_B_eye_left
        mask_B["mask_B_eye_right"] = mask_B_eye_right
        mask_B["index_B_eye_left"] = index_B_eye_left
        mask_B["index_B_eye_right"] = index_B_eye_right
        mask_B["mask_B_skin"] = mask_B_skin
        mask_B["index_B_skin"] = index_B_skin
        mask_B["mask_B_lip"] = mask_B_lip
        mask_B["index_B_lip"] = index_B_lip

        return {
            "nonmakeup_seg": nonmakeup_seg,
            "makeup_seg": makeup_seg,
            "nonmakeup_img": nonmakeup_img,
            "makeup_img": makeup_img,
            "mask_A": mask_A,
            "mask_B": mask_B,
            "makeup_unchanged": makeup_unchanged,
            "nonmakeup_unchanged": nonmakeup_unchanged,
        }

    def rectangle_box(self, mask_A, mask_B, mask_A_face):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)
        mask_A_face = mask_A_face.unsqueeze(0)

        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[
            :, :, min(x_A_index) - 5 : max(x_A_index) + 6, min(y_A_index) - 5 : max(y_A_index) + 6
        ] = mask_A_face[:, :, min(x_A_index) - 5 : max(x_A_index) + 6, min(y_A_index) - 5 : max(y_A_index) + 6]
        mask_B_temp[
            :, :, min(x_B_index) - 5 : max(x_B_index) + 6, min(y_B_index) - 5 : max(y_B_index) + 6
        ] = mask_A_face[:, :, min(x_B_index) - 5 : max(x_B_index) + 6, min(y_B_index) - 5 : max(y_B_index) + 6]

        mask_A_temp = mask_A_temp.squeeze(0)
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        mask_A_face = mask_A_face.squeeze(0)
        mask_B_temp = mask_B_temp.squeeze(0)

        return mask_A_temp, mask_B_temp

    def dilate_box(self, mask_A, mask_B, mask_A_face):
        raise NotImplementedError()

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        if self.eye_box in ("rectangle", "rect", "r"):
            return self.rectangle_box(mask_A, mask_B, mask_A_face)
        elif self.eye_box in ("dilate", "d"):
            return self.dilate_box(mask_A, mask_B, mask_A_face)

    def mask_preprocess(self, mask_A, mask_B):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)

        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]

        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]

        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]

        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)

        return mask_A, mask_B, index, index_2


class TransferDataset(SCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.makeup_names)

    def pick(self, index):
        makeup_name = self.makeup_names[index]

        idx = np.random.randint(0, len(self.non_makeup_names))
        nonmakeup_name = self.non_makeup_names[idx]

        return nonmakeup_name, makeup_name
