import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt


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
    img = img.transpose(0, 1).transpose(0, 2).contiguous()

    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class TransferDataset:
    def __init__(self, source_path, reference_path, source_seg_path, reference_seg_path, opt):
        self.random = None
        self.opt = opt
        self.n_components = opt.n_components

        self.source_path = source_path
        self.reference_path = reference_path

        self.source_seg_path = source_seg_path
        self.reference_seg_path = reference_seg_path

        self.transform = transforms.Compose(
            [
                transforms.Resize((opt.img_size, opt.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.transform_mask = transforms.Compose(
            [transforms.Resize((opt.img_size, opt.img_size), interpolation=InterpolationMode.NEAREST), ToTensor]
        )

    def __getitem__(self, index):
        source_img = self.transform(Image.open(self.source_path).convert("RGB"))
        reference_img = self.transform(Image.open(self.reference_path).convert("RGB"))

        source_seg_img = Image.open(self.source_seg_path)
        reference_seg_img = Image.open(self.reference_seg_path)

        mask_A = self.transform_mask(source_seg_img)  # source
        mask_B = self.transform_mask(reference_seg_img)  # reference

        # plt.imshow(mask_A.permute(1, 2, 0))
        # plt.show()

        makeup_seg = torch.zeros([self.n_components, self.opt.img_size, self.opt.img_size], dtype=torch.float)
        nonmakeup_seg = torch.zeros([self.n_components, self.opt.img_size, self.opt.img_size], dtype=torch.float)
        makeup_unchanged = (
            (mask_B == 7).float()
            + (mask_B == 2).float()
            + (mask_B == 6).float()
            + (mask_B == 1).float()
            + (mask_B == 11).float()
        )
        nonmakeup_unchanged = (
            (mask_A == 7).float()
            + (mask_A == 2).float()
            + (mask_A == 6).float()
            + (mask_A == 1).float()
            + (mask_A == 11).float()
        )

        # plt.imshow(makeup_unchanged.permute(1, 2, 0))
        # plt.show()
        # plt.imshow(nonmakeup_unchanged.permute(1, 2, 0))
        # plt.show()

        mask_A_lip = (mask_A == 9).float() + (mask_A == 13).float()
        mask_B_lip = (mask_B == 9).float() + (mask_B == 13).float()
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
        makeup_seg[0] = mask_B_lip
        nonmakeup_seg[0] = mask_A_lip
        mask_A_skin = (mask_A == 4).float() + (mask_A == 8).float() + (mask_A == 10).float()
        mask_B_skin = (mask_B == 4).float() + (mask_B == 8).float() + (mask_B == 10).float()
        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
        makeup_seg[1] = mask_B_skin
        nonmakeup_seg[1] = mask_A_skin
        mask_A_eye_left = (mask_A == 6).float()
        mask_A_eye_right = (mask_A == 1).float()
        mask_B_eye_left = (mask_B == 6).float()
        mask_B_eye_right = (mask_B == 1).float()
        mask_A_face = (mask_A == 4).float() + (mask_A == 8).float()
        mask_B_face = (mask_B == 4).float() + (mask_B == 8).float()
        # avoid the es of ref are closed
        if not ((mask_B_eye_left > 0).any() and (mask_B_eye_right > 0).any()):
            return {}
        # mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)

        # plt.imshow((mask_B_eye_left + mask_B_eye_right).permute(1, 2, 0))
        # plt.title("Before rebound")
        # plt.show()

        # mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)

        # plt.imshow((mask_B_eye_left + mask_B_eye_right).permute(1, 2, 0))
        # plt.title("rebound")
        # plt.show()

        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(
            mask_A_eye_left, mask_B_eye_left
        )
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(
            mask_A_eye_right, mask_B_eye_right
        )

        nonmakeup_seg[2] = mask_A_eye_left + mask_A_eye_right
        makeup_seg[2] = mask_B_eye_left + mask_B_eye_right

        plt.imshow(makeup_seg[2])
        plt.show()
        plt.imshow(nonmakeup_seg[2])
        plt.show()

        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["index_A_eye_left"] = index_A_eye_left
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
            "nonmakeup_img": source_img,
            "makeup_img": reference_img,
            "mask_A": mask_A,
            "mask_B": mask_B,
            "makeup_unchanged": makeup_unchanged,
            "nonmakeup_unchanged": nonmakeup_unchanged,
        }

    def __len__(self):
        return 1

    def name(self):
        return "TransferDataset"

    def rebound_box(self, mask_A, mask_B, mask_A_face):
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

        A_limits = (0, 0, 0, 0)
        B_limits = (0, 0, 0, 0)

        mask_A_temp[
            :,
            :,
            min(x_A_index) - A_limits[0] : max(x_A_index) + A_limits[1],
            min(y_A_index) - A_limits[2] : max(y_A_index) + A_limits[3],
        ] = mask_A_face[
            :,
            :,
            min(x_A_index) - A_limits[0] : max(x_A_index) + A_limits[1],
            min(y_A_index) - A_limits[2] : max(y_A_index) + A_limits[3],
        ]
        mask_B_temp[
            :,
            :,
            min(x_B_index) - B_limits[0] : max(x_B_index) + B_limits[1],
            min(y_B_index) - B_limits[2] : max(y_B_index) + B_limits[3],
        ] = mask_A_face[
            :,
            :,
            min(x_B_index) - B_limits[0] : max(x_B_index) + B_limits[1],
            min(y_B_index) - B_limits[2] : max(y_B_index) + B_limits[3],
        ]
        mask_A_temp = mask_A_temp.squeeze(0)
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        mask_A_face = mask_A_face.squeeze(0)
        mask_B_temp = mask_B_temp.squeeze(0)

        # plt.imshow(mask_A.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        # plt.show()

        # plt.imshow(mask_A_temp.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        # plt.show()

        # plt.imshow(mask_B_temp.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        # plt.show()

        return mask_A_temp, mask_B_temp

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

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()

        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)


class TransferDataLoader:
    def __init__(self, source_path, reference_path, source_seg_path, reference_seg_path, opt):
        self.dataset = TransferDataset(source_path, reference_path, source_seg_path, reference_seg_path, opt)
        self.dataloader = DataLoader(
            self.dataset, batch_size=opt.batchSize, shuffle=not opt.serial_batches, num_workers=int(opt.nThreads)
        )

    def name(self):
        return "TransferDataLoader"

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
