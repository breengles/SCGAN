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
        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(
            mask_A_eye_left, mask_B_eye_left
        )
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(
            mask_A_eye_right, mask_B_eye_right
        )
        makeup_seg[2] = mask_B_eye_left + mask_B_eye_right

        plt.imshow(makeup_seg[2].cpu().numpy().astype(np.uint8))
        plt.show()

        nonmakeup_seg[2] = mask_A_eye_left + mask_A_eye_right

        plt.imshow(nonmakeup_seg[2].cpu().numpy().astype(np.uint8))
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


if __name__ == "__main__":

    class Options:
        def __init__(
            self,
            phase="train",  # use test for inference
            beta1=0.5,
            beta2=0.999,
            g_lr=2e-4,
            d_lr=2e-4,
            lambda_A=10.0,
            lambda_B=10.0,
            lambda_idt=0.5,
            lambda_his_lip=1.0,
            lambda_his_skin=0.1,
            lambda_his_eye=1.0,
            lambda_vgg=5e-3,
            num_epochs=100,
            epochs_decay=0,
            g_step=1,
            log_step=8,
            save_step=2048,
            snapshot_path="./checkpoints/",
            save_path="./results/",
            snapshot_step=10,
            perceptual_layers=3,
            partial=False,
            interpolation=False,
            init_type="xavier",
            dataroot="MT-Dataset/images",  # folder with
            dirmap="MT-Dataset/parsing",
            batchSize=1,
            input_nc=3,
            img_size=256,
            output_nc=3,
            d_conv_dim=64,
            d_repeat_num=3,
            ngf=64,
            gpu_ids="0",
            nThreads=2,
            norm1="SN",
            serial_batches=True,
            n_components=3,
            n_res=3,
            padding_type="reflect",
            use_flip=0,
            n_downsampling=2,
            style_dim=192,
            mlp_dim=256,
        ):
            self.phase = phase
            self.beta1 = beta1
            self.beta2 = beta2
            self.g_lr = g_lr
            self.d_lr = d_lr
            self.lambda_A = lambda_A
            self.lambda_B = lambda_B
            self.lambda_idt = lambda_idt
            self.lambda_his_lip = lambda_his_lip
            self.lambda_his_skin = lambda_his_skin
            self.lambda_his_eye = lambda_his_eye
            self.lambda_vgg = lambda_vgg
            self.num_epochs = num_epochs
            self.epochs_decay = epochs_decay
            self.g_step = g_step
            self.log_step = log_step
            self.save_step = save_step
            self.snapshot_path = snapshot_path
            self.save_path = save_path
            self.snapshot_step = snapshot_step
            self.perceptual_layers = perceptual_layers
            self.partial = partial
            self.interpolation = interpolation

            self.init_type = init_type
            self.dataroot = dataroot
            self.dirmap = dirmap
            self.batchSize = batchSize
            self.input_nc = input_nc
            self.img_size = img_size
            self.output_nc = output_nc
            self.d_conv_dim = d_conv_dim
            self.d_repeat_num = d_repeat_num
            self.ngf = ngf
            self.gpu_ids = gpu_ids
            self.nThreads = nThreads
            self.norm1 = norm1
            self.serial_batches = serial_batches
            self.n_components = n_components
            self.n_res = n_res
            self.padding_type = padding_type
            self.use_flip = use_flip
            self.n_downsampling = n_downsampling
            self.style_dim = style_dim
            self.mlp_dim = mlp_dim

    opt = Options(phase="test", dataroot="../dataset2/images", dirmap="../dataset2/parsing", save_path="../results/")

    source_path = "../dataset/non-makeup/00313.jpg"
    reference_path = "../dataset/makeup/b7fd5266d01609248414333cdf0735fae6cd34e7.png"

    source_seg_path = "../dataset/non-makeup/parsing/00313.jpg"
    ref_seg_path = "../dataset/makeup/parsing/b7fd5266d01609248414333cdf0735fae6cd34e7.png"

    dataloader = TransferDataLoader(source_path, reference_path, source_seg_path, ref_seg_path, opt=opt)

    for batch in dataloader:
        print("ololo")
