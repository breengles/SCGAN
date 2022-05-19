import os.path
from enum import IntEnum

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import hflip


class Regions(IntEnum):
    """
    commented -- original SCGAN labels
    used -- labels from faceparsing
    """

    FACE = 1  # 4
    NOSE = 10  # 8
    RIGHT_EYE = 5  # 1
    LEFT_EYE = 4  # 6
    UPPER_LIP_VERMILLION = 12  # 9
    LOWER_LIP_VERMILLION = 13  # 13
    RIGHT_EYEBROW = 3  # 2
    LEFT_EYEBROW = 2  # 7
    TEETH = 11  # 11
    NECK = 14  # 10
    BACKGROUND = 0  # 0
    HAIR = 17  # 12


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
    def __init__(self, dataroot, img_size, n_components, eye_box="rectangle", resize=True):
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

        transform = []
        transform_mask = []

        if resize:
            transform.append(transforms.Resize((img_size, img_size)))
            transform_mask.append(transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST))

        transform.extend([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        transform_mask.append(ToTensor)

        self.transform = transforms.Compose(transform)
        self.transform_mask = transforms.Compose(transform_mask)

        self.flip_proba = 0.5
        self.dilation_kernel = np.ones((3, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.makeup_names)

    def pick(self, index):
        makeup_name = self.makeup_names[index]

        nonmakeup_index = torch.randint(low=0, high=len(self.non_makeup_names), size=(1,))
        nonmakeup_name = self.non_makeup_names[nonmakeup_index]

        return nonmakeup_name, makeup_name

    def get_nonmakeup_img_seg(self, nonmakeup_name):
        nonmakeup_path = os.path.join(self.dir_img_nonmakeup, nonmakeup_name)
        nonmakeup_img = Image.open(nonmakeup_path).convert("RGB")
        nonmakeup_seg_img = Image.open(os.path.join(self.dir_seg_nonmakeup, nonmakeup_name))
        nonmakeup_img = self.transform(nonmakeup_img)

        mask_A = self.transform_mask(nonmakeup_seg_img)  # nonmakeup
        
        if torch.rand(1) < self.flip_proba:
            nonmakeup_img = hflip(nonmakeup_img)
            mask_A = hflip(mask_A)

        nonmakeup_seg = torch.zeros([self.n_components, self.img_size, self.img_size], dtype=torch.float)
        
        mask_A_lip = (mask_A == Regions.UPPER_LIP_VERMILLION).float() + (mask_A == Regions.LOWER_LIP_VERMILLION).float()
        mask_A_skin = (
            (mask_A == Regions.FACE).float() + (mask_A == Regions.NOSE).float() + (mask_A == Regions.NECK).float()
        )
        mask_A_eye_left = (mask_A == Regions.LEFT_EYE).float()
        mask_A_eye_right = (mask_A == Regions.RIGHT_EYE).float()
        mask_A_face = (mask_A == Regions.FACE).float() + (mask_A == Regions.NOSE).float()
        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)

        nonmakeup_seg[0] = mask_A_lip
        nonmakeup_seg[1] = mask_A_skin
        nonmakeup_seg[2] = mask_A_eye_left + mask_A_eye_right
        
        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["mask_A_skin"] = mask_A_skin
        mask_A["mask_A_lip"] = mask_A_lip
        
        return nonmakeup_img, nonmakeup_seg, mask_A


    def get_makeup_img_seg(self, makeup_name):
        makeup_path = os.path.join(self.dir_img_makeup, makeup_name)
        makeup_img = Image.open(makeup_path).convert("RGB")
        makeup_seg_img = Image.open(os.path.join(self.dir_seg_makeup, makeup_name))
        makeup_img = self.transform(makeup_img)
        mask_B = self.transform_mask(makeup_seg_img)  # makeup
        
        if torch.rand(1) < self.flip_proba:
            makeup_img = hflip(makeup_img)
            mask_B = hflip(mask_B)
            
        makeup_seg = torch.zeros([self.n_components, self.img_size, self.img_size], dtype=torch.float)
        
        mask_B_lip = (mask_B == Regions.UPPER_LIP_VERMILLION).float() + (mask_B == Regions.LOWER_LIP_VERMILLION).float()
        
        mask_B_skin = (
            (mask_B == Regions.FACE).float() + (mask_B == Regions.NOSE).float() + (mask_B == Regions.NECK).float()
        )
        
        mask_B_eye_left = (mask_B == Regions.LEFT_EYE).float()
        mask_B_eye_right = (mask_B == Regions.RIGHT_EYE).float()
        
        
        mask_B_face = (mask_B == Regions.FACE).float() + (mask_B == Regions.NOSE).float()
            
        # avoid the es of ref are closed
        if not ((mask_B_eye_left > 0).any() and (mask_B_eye_right > 0).any()):
            valid = False
        else:
            valid = True
            
        if valid:
            mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
        
        makeup_seg[0] = mask_B_lip
        makeup_seg[1] = mask_B_skin
        makeup_seg[2] = mask_B_eye_left + mask_B_eye_right
        
        mask_B = {}
        mask_B["mask_B_eye_left"] = mask_B_eye_left
        mask_B["mask_B_eye_right"] = mask_B_eye_right
        mask_B["mask_B_skin"] = mask_B_skin
        mask_B["mask_B_lip"] = mask_B_lip
        
        return makeup_img, makeup_seg, mask_B, valid
        

    def __getitem__(self, index):
        nonmakeup_name, makeup_name = self.pick(index)
        
        nonmakeup_img, nonmakeup_seg, mask_A = self.get_nonmakeup_img_seg(nonmakeup_name)
        makeup_img, makeup_seg, mask_B, valid = self.get_makeup_img_seg(makeup_name)

        # makeup_unchanged = (
        #     (mask_B == Regions.LEFT_EYEBROW).float()
        #     + (mask_B == Regions.RIGHT_EYEBROW).float()
        #     + (mask_B == Regions.LEFT_EYE).float()
        #     + (mask_B == Regions.RIGHT_EYE).float()
        #     + (mask_B == Regions.TEETH).float()
        # )
        # nonmakeup_unchanged = (
        #     (mask_A == Regions.LEFT_EYEBROW).float()
        #     + (mask_A == Regions.RIGHT_EYEBROW).float()
        #     + (mask_A == Regions.LEFT_EYE).float()
        #     + (mask_A == Regions.RIGHT_EYE).float()
        #     + (mask_A == Regions.TEETH).float()
        # )

        return {
            "nonmakeup_seg": nonmakeup_seg,
            "makeup_seg": makeup_seg,
            "nonmakeup_img": nonmakeup_img,
            "makeup_img": makeup_img,
            "mask_A": mask_A,
            "mask_B": mask_B,
            # "makeup_unchanged": makeup_unchanged,
            # "nonmakeup_unchanged": nonmakeup_unchanged,
            "valid": valid,
        }

    def rectangle_box(self, mask_right_eye, mask_left_eye, mask_face):
        mask_right_eye = mask_right_eye.unsqueeze(0)
        mask_left_eye = mask_left_eye.unsqueeze(0)
        mask_face = mask_face.unsqueeze(0)

        index_tmp = torch.nonzero(mask_right_eye, as_tuple=False)
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]

        index_tmp = torch.nonzero(mask_left_eye, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]

        mask_right_eye_temp = mask_right_eye.copy_(mask_right_eye)
        mask_left_eye_temp = mask_left_eye.copy_(mask_left_eye)

        mask_right_eye_temp[
            :, :, min(x_A_index) - 5 : max(x_A_index) + 6, min(y_A_index) - 5 : max(y_A_index) + 6
        ] = mask_face[:, :, min(x_A_index) - 5 : max(x_A_index) + 6, min(y_A_index) - 5 : max(y_A_index) + 6]
        mask_left_eye_temp[
            :, :, min(x_B_index) - 5 : max(x_B_index) + 6, min(y_B_index) - 5 : max(y_B_index) + 6
        ] = mask_face[:, :, min(x_B_index) - 5 : max(x_B_index) + 6, min(y_B_index) - 5 : max(y_B_index) + 6]

        mask_right_eye_temp = mask_right_eye_temp.squeeze(0)
        mask_right_eye = mask_right_eye.squeeze(0)
        mask_left_eye = mask_left_eye.squeeze(0)
        mask_face = mask_face.squeeze(0)
        mask_left_eye_temp = mask_left_eye_temp.squeeze(0)

        return mask_right_eye_temp, mask_left_eye_temp

    def _extract_cnt(self, mask_eye, mask_face, iterations, thickness):
        dilated_eye = cv2.dilate(mask_eye, self.dilation_kernel, iterations=iterations)
        cnt_eye, _ = cv2.findContours(dilated_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_mask_eye = np.zeros_like(dilated_eye)
        cnt_mask_eye = cv2.drawContours(cnt_mask_eye, cnt_eye, -1, (128, 128, 128), thickness).astype(bool)

        mask_eye_out = np.zeros_like(mask_eye)
        mask_eye_out[cnt_mask_eye] = mask_face[cnt_mask_eye]
        return mask_eye_out

    def dilate_box(self, mask_right_eye, mask_left_eye, mask_face, thickness=None, iterations=3):
        mask_right_eye_tmp = mask_right_eye.clone().permute(1, 2, 0).numpy().astype(np.uint8)
        mask_left_eye_tmp = mask_left_eye.clone().permute(1, 2, 0).numpy().astype(np.uint8)
        mask_face_tmp = mask_face.clone().permute(1, 2, 0).numpy().astype(np.uint8)

        if thickness is None:
            thickness = int(1 + max(mask_face.shape[0], mask_face.shape[1]) / 100 * 15)  # 15% contour

        mask_right_eye_out = self._extract_cnt(mask_right_eye_tmp, mask_face_tmp, iterations, thickness)
        mask_left_eye_out = self._extract_cnt(mask_left_eye_tmp, mask_face_tmp, iterations, thickness)

        mask_right_eye_out = torch.from_numpy(mask_right_eye_out).permute(2, 0, 1).contiguous()
        mask_left_eye_out = torch.from_numpy(mask_left_eye_out).permute(2, 0, 1).contiguous()

        return mask_right_eye_out, mask_left_eye_out

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        if self.eye_box in ("rectangle", "rect", "r"):
            return self.rectangle_box(mask_A, mask_B, mask_A_face)
        elif self.eye_box in ("dilate", "d"):
            return self.dilate_box(mask_A, mask_B, mask_A_face)


class TransferDataset(SCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip_proba = 0

    def __len__(self):
        return len(self.makeup_names) * len(self.non_makeup_names)

    def get_nonmakeup_img_seg(self, nonmakeup_name):
        nonmakeup_path = os.path.join(self.dir_img_nonmakeup, nonmakeup_name)
        nonmakeup_img = Image.open(nonmakeup_path).convert("RGB")
        nonmakeup_img = self.transform(nonmakeup_img)

        return nonmakeup_img, torch.zeros_like(nonmakeup_img), {}

    def pick(self, index):
        makeup_idx = index % len(self.non_makeup_names)
        nonmakeup_idx = index // len(self.makeup_names)

        makeup_name = self.makeup_names[makeup_idx]
        nonmakeup_name = self.non_makeup_names[nonmakeup_idx]

        return nonmakeup_name, makeup_name
