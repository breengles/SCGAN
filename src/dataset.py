import os
from enum import IntEnum, auto

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


class EyeBox(IntEnum):
    DILATE = auto()
    RECTANGLE = auto()


def determine_eye_box(eye_box):
    eye_box = eye_box.lower()
    if eye_box in ("d", "dilate"):
        return EyeBox.DILATE
    elif eye_box in ("r", "rect", "rectangle"):
        return EyeBox.RECTANGLE
    else:
        raise ValueError(f"Unknown eye box {eye_box}")


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
    """
    handle PIL Image for segmentation mask
    """

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

    img = img.reshape(pic.size[1], pic.size[0], nchannel)

    # put it from HWC to CHW format
    img = img.permute(2, 0, 1).contiguous()

    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class SCDataset(Dataset):
    def __init__(
        self,
        dataroot="dataset/",
        img_size=512,
        n_components=3,
        eye_box=EyeBox.DILATE,
        dilation_kernel=np.ones((3, 3), dtype=np.uint8),
    ):
        self.dataroot = dataroot

        self.images_path = os.path.join(self.dataroot, "images")
        self.makeup_img_path = os.path.join(self.images_path, "makeup")
        self.nonmakeup_img_path = os.path.join(self.images_path, "non-makeup")

        self.makeup_img_names = os.listdir(self.makeup_img_path)
        self.nonmakeup_img_names = os.listdir(self.nonmakeup_img_path)

        self.segments_path = os.path.join(self.dataroot, "segments")
        self.makeup_seg_path = os.path.join(self.segments_path, "makeup")
        self.nonmakeup_seg_path = os.path.join(self.segments_path, "non-makeup")

        self.eye_box = eye_box
        self.dilation_kernel = dilation_kernel

        self.img_transform = T.Compose(
            [T.ToTensor(), T.Resize((img_size, img_size)), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.seg_transform = T.Compose(
            [ToTensor, T.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)]
        )

        self.n_components = n_components
        self.img_size = img_size
        self.pad_size = img_size * img_size

        self.random = None

    def __len__(self):
        return len(os.listdir(self.nonmakeup_img_path))

    def _dilate_eye(self, segment, part=Regions.LEFT_EYE, iterations=1, thickness=None):
        if not (segment == part.value).any():
            return segment

        init_segment = segment.permute(1, 2, 0).clone()
        out_segment = init_segment.clone()

        if thickness is None:
            thickness = int(1 + max(segment.shape[0], segment.shape[1]) / 100 * 15)  # 15% contour

        tmp = np.zeros(init_segment.shape, dtype=np.uint8)
        tmp[init_segment == part.value] = 1

        dilated = cv2.dilate(tmp, self.dilation_kernel, iterations=iterations)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = np.zeros_like(dilated)
        cnt = cv2.drawContours(cnt, contours, -1, (128, 128, 128), thickness).astype(bool)

        out_segment[cnt] = part.value

        # rewrite boundaries
        for p in (Regions.BACKGROUND, Regions.RIGHT_EYEBROW, Regions.LEFT_EYEBROW, Regions.NOSE, Regions.HAIR):
            out_segment[init_segment == p.value] = p.value

        out_segment[init_segment == part.value] = 0  # cut initial eye out

        return out_segment.permute(2, 0, 1).contiguous()

    @staticmethod
    def _rectangle_eye(segment, part):
        if not (segment == part.value).any():
            return segment

        init_segment = segment.clone()
        out_segment = segment.clone()

        tmp = torch.zeros_like(out_segment)
        tmp[out_segment == part.value] = 1
        index_tmp = torch.nonzero(tmp, as_tuple=False)

        x_index = index_tmp[:, 1]
        y_index = index_tmp[:, 2]

        out_segment[:, min(x_index) - 5 : max(x_index) + 6, min(y_index) - 5 : max(y_index) + 6] = part.value

        out_segment[init_segment == part.value] = 0  # cut initial eye out

        return out_segment

    def rebound_eyes(self, seg):
        for part in (Regions.LEFT_EYE, Regions.RIGHT_EYE):
            if self.eye_box == EyeBox.DILATE:
                seg = self._dilate_eye(seg, part, iterations=3)
            elif self.eye_box == EyeBox.RECTANGLE:
                seg = self._rectangle_eye(seg, part)
            else:
                raise ValueError(f"Unknown eye box {self.eye_box}")

        return seg

    @staticmethod
    def _get_mask_unchanged(mask):
        return (
            (mask == Regions.LEFT_EYEBROW.value)
            + (mask == Regions.RIGHT_EYEBROW.value)
            + (mask == Regions.LEFT_EYE.value)
            + (mask == Regions.RIGHT_EYE.value)
            + (mask == Regions.TEETH.value)
        ).float()

    @staticmethod
    def _get_mask_lip(mask):
        return ((mask == Regions.UPPER_LIP_VERMILLION.value) + (mask == Regions.LOWER_LIP_VERMILLION.value)).float()

    @staticmethod
    def _get_mask_skin(mask):
        return ((mask == Regions.FACE.value) + (mask == Regions.NOSE.value) + (mask == Regions.NECK.value)).float()

    @staticmethod
    def _get_mask_eye_left(mask):
        return (mask == Regions.LEFT_EYE.value).float()

    @staticmethod
    def _get_mask_eye_right(mask):
        return (mask == Regions.RIGHT_EYE.value).float()

    @staticmethod
    def _get_mask_face(mask):
        return ((mask == Regions.FACE.value) + (mask == Regions.NOSE.value)).float()

    def _get_regions(self, mask):
        mask_lip = self._get_mask_lip(mask)
        mask_skin = self._get_mask_skin(mask)
        mask_eye_left = self._get_mask_eye_left(mask)
        mask_eye_right = self._get_mask_eye_right(mask)
        mask_face = self._get_mask_face(mask)

        return mask_lip, mask_skin, mask_face, mask_eye_left, mask_eye_right

    def _get_indices(self, mask):
        index = torch.nonzero(mask, as_tuple=False)
        x_index = index[:, 2]
        y_index = index[:, 3]

        x_index = F.pad(x_index, (0, self.pad_size - x_index.shape[0]), value=-1)
        y_index = F.pad(y_index, (0, self.pad_size - y_index.shape[0]), value=-1)

        return x_index, y_index

    def _get_masks_index(self, mask_A, mask_B):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)

        x_index_A, y_index_A = self._get_indices(mask_A)
        x_index_B, y_index_B = self._get_indices(mask_B)

        index_1 = [x_index_A, y_index_A, x_index_B, y_index_B]
        index_2 = [x_index_B, y_index_B, x_index_A, y_index_A]

        return torch.vstack(index_1), torch.vstack(index_2)

    def pick_nonmakeup_name(self, index):
        return self.nonmakeup_img_names[index]

    def pick_makeup_name(self):
        idx = np.random.randint(low=0, high=len(self.makeup_img_names), size=1)[0]
        return self.makeup_img_names[idx]

    def __getitem__(self, index):
        makeup_img_name = self.pick_makeup_name()  # random pick
        nonmakeup_img_name = self.pick_nonmakeup_name(index)  # pick by index

        makeup_img_path = os.path.join(self.makeup_img_path, makeup_img_name)
        nonmakeup_img_path = os.path.join(self.nonmakeup_img_path, nonmakeup_img_name)
        makeup_seg_path = os.path.join(self.makeup_seg_path, makeup_img_name)
        nonmakeup_seg_path = os.path.join(self.nonmakeup_seg_path, nonmakeup_img_name)

        makeup_img = Image.open(makeup_img_path).convert("RGB")
        nonmakeup_img = Image.open(nonmakeup_img_path).convert("RGB")
        makeup_seg = Image.open(makeup_seg_path)
        nonmakeup_seg = Image.open(nonmakeup_seg_path)

        makeup_img = self.img_transform(makeup_img)
        nonmakeup_img = self.img_transform(nonmakeup_img)
        makeup_seg = self.seg_transform(makeup_seg)
        nonmakeup_seg = self.seg_transform(nonmakeup_seg)

        # makeup_mask_unchanged = self._get_mask_unchanged(makeup_seg)
        # nonmakeup_mask_unchanged = self._get_mask_unchanged(nonmakeup_seg)

        # fig, ax = plt.subplots(2, 3)
        #
        # ax[0][0].imshow(nonmakeup_img.permute(1, 2, 0))
        # ax[1][0].imshow(makeup_img.permute(1, 2, 0))
        #
        # ax[0][1].imshow(nonmakeup_seg.permute(1, 2, 0))
        # ax[1][1].imshow(makeup_seg.permute(1, 2, 0))

        makeup_seg = self.rebound_eyes(makeup_seg)
        nonmakeup_seg = self.rebound_eyes(nonmakeup_seg)

        # ax[0][2].imshow(nonmakeup_seg.permute(1, 2, 0))
        # ax[1][2].imshow(makeup_seg.permute(1, 2, 0))
        #
        # plt.show()
        # exit()

        (
            makeup_mask_lip,
            makeup_mask_skin,
            makeup_mask_face,
            makeup_mask_left_eye,
            makeup_mask_right_eye,
        ) = self._get_regions(makeup_seg)
        (
            nonmakeup_mask_lip,
            nonmakeup_mask_skin,
            nonmakeup_mask_face,
            nonmakeup_mask_left_eye,
            nonmakeup_mask_right_eye,
        ) = self._get_regions(nonmakeup_seg)

        # nonmakeup_lip_index, makeup_lip_index = self._get_masks_index(nonmakeup_mask_lip, makeup_mask_lip)
        # nonmakeup_skin_index, makeup_skin_index = self._get_masks_index(nonmakeup_mask_skin, makeup_mask_skin)
        # nonmakeup_left_eye_index, makeup_left_eye_index = self._get_masks_index(
        #     nonmakeup_mask_left_eye, makeup_mask_left_eye
        # )
        # nonmakeup_right_eye_index, makeup_right_eye_index = self._get_masks_index(
        #     nonmakeup_mask_right_eye, makeup_mask_right_eye
        # )

        mask_makeup = torch.zeros([self.n_components, self.img_size, self.img_size], dtype=torch.float32)
        mask_nonmakeup = torch.zeros([self.n_components, self.img_size, self.img_size], dtype=torch.float32)

        mask_makeup[0] = makeup_mask_lip
        mask_makeup[1] = makeup_mask_skin
        mask_makeup[2] = makeup_mask_left_eye + makeup_mask_right_eye

        mask_nonmakeup[0] = nonmakeup_mask_lip
        mask_nonmakeup[1] = nonmakeup_mask_skin
        mask_nonmakeup[2] = nonmakeup_mask_left_eye + nonmakeup_mask_right_eye

        return {
            "nonmakeup_seg": mask_nonmakeup,
            "makeup_seg": mask_makeup,
            "nonmakeup_img": nonmakeup_img,
            "makeup_img": makeup_img,
            # "makeup_unchanged": makeup_mask_unchanged,
            # "nonmakeup_unchanged": nonmakeup_mask_unchanged,
            "makeup_mask_lip": makeup_mask_lip,
            "makeup_mask_skin": makeup_mask_skin,
            "makeup_mask_left_eye": makeup_mask_left_eye,
            "makeup_mask_right_eye": makeup_mask_right_eye,
            "nonmakeup_mask_lip": nonmakeup_mask_lip,
            "nonmakeup_mask_skin": nonmakeup_mask_skin,
            "nonmakeup_mask_left_eye": nonmakeup_mask_left_eye,
            "nonmakeup_mask_right_eye": nonmakeup_mask_right_eye,
            # "makeup_lip_index": makeup_lip_index,
            # "makeup_skin_index": makeup_skin_index,
            # "makeup_left_eye_index": makeup_left_eye_index,
            # "makeup_right_eye_index": makeup_right_eye_index,
            # "nonmakeup_lip_index": nonmakeup_lip_index,
            # "nonmakeup_skin_index": nonmakeup_skin_index,
            # "nonmakeup_left_eye_index": nonmakeup_left_eye_index,
            # "nonmakeup_right_eye_index": nonmakeup_right_eye_index,
            "valid": ((makeup_mask_left_eye > 0).any() and (makeup_mask_right_eye > 0).any()),
        }


if __name__ == "__main__":
    dataset = SCDataset(
        "/home/breengles/Dropbox/projects/MakeupScience/software/SCGAN/datasets/non-dilated/daniil.128px.cut.crop.overfit",
        img_size=128,
    )

    dataloader = DataLoader(dataset, batch_size=4)

    for idx, item in enumerate(dataloader):
        print(item["mask_nonmakeup"].shape)
        print(item["mask_makeup"].shape)
        print(item["nonmakeup_img"].shape)
        print(item["makeup_img"].shape)

        for k, v in item["makeup_regions"].items():
            if isinstance(v, list):
                for x in v:
                    print(x.shape)
            else:
                print(v.shape)

        for k, v in item["nonmakeup_regions"].items():
            if isinstance(v, list):
                for x in v:
                    print(x.shape)
            else:
                print(v.shape)

        print(item["makeup_unchanged"].shape)
        print(item["nonmakeup_unchanged"].shape)
        print(item["valid"].shape)
