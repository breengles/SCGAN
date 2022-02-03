import os
from tkinter import Variable

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage

from faceparsing.test import evaluate_image
from models.models import create_model


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


def handle_parsing(seg):
    """
    To match SCGAN segmentation
    """
    new = np.zeros_like(seg)
    new[seg == 0] = 0
    new[seg == 1] = 4
    new[seg == 2] = 7
    new[seg == 3] = 2
    new[seg == 4] = 6
    new[seg == 5] = 1
    new[seg == 6] = 0
    new[seg == 7] = 3
    new[seg == 8] = 5
    new[seg == 9] = 0
    new[seg == 10] = 8
    new[seg == 11] = 11
    new[seg == 12] = 9
    new[seg == 13] = 13
    new[seg == 14] = 10
    new[seg == 15] = 0
    new[seg == 16] = 0
    new[seg == 17] = 12
    return new


def dilate(img, kernel, iterations=1, part=1):
    tmp = np.zeros_like(img)

    mask = img == part
    tmp[mask] = 1
    out = cv2.dilate(tmp, kernel, iterations=iterations)

    out = out.astype(bool)
    img[out] = part

    return img


def get_boundary_points(img):
    cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt_max = 0
    for c in cnt:
        if c.shape[0] > cnt_max:
            cnt_max = c.shape[0]
            cnt_ = c

    cnt = cnt_.squeeze()
    left = cnt[:, 0].argmin()
    right = cnt[:, 0].argmax()
    upper = cnt[:, 1].argmin()
    lower = cnt[:, 1].argmax()

    return (cnt[left], cnt[right], cnt[upper], cnt[lower])


def fit_linear_eye2brow(eye_points, brow_points):
    xs_left = [eye_points[0][0], brow_points[0][0]]
    xs_right = [eye_points[1][0], brow_points[1][0]]

    ys_left = [eye_points[0][1], brow_points[0][1]]
    ys_right = [eye_points[1][1], brow_points[1][1]]

    y_left = np.polynomial.polynomial.Polynomial.fit(ys_left, xs_left, deg=1)
    y_right = np.polynomial.polynomial.Polynomial.fit(ys_right, xs_right, deg=1)

    return y_left, y_right, (xs_left, ys_left), (xs_right, ys_right)


def fill_between_eye_eyebrow(source_mask, part=1):
    eye = np.zeros_like(source_mask)
    brow = np.zeros_like(source_mask)

    eye[source_mask == part] = 1
    brow[source_mask == part + 1] = 1

    eye_points = get_boundary_points(eye)
    brow_points = get_boundary_points(brow)

    y_left, y_right, p_left, p_right = fit_linear_eye2brow(eye_points, brow_points)

    width = source_mask.shape[0]
    height = source_mask.shape[1]
    area = np.repeat(np.arange(start=height - 1, stop=-1, step=-1), width).reshape(height, width).astype(int)

    ys_idx = np.arange(area.shape[1])

    area_lowest = area - y_left(ys_idx)
    area_highest = area - y_right(ys_idx)

    bottom_mask = area_lowest < 0
    upper_mask = area_highest > 0

    y_mask = ~(bottom_mask + upper_mask)
    x_mask = np.zeros_like(y_mask)
    x_mask[:, min(*p_left[1], *p_right[1]) : max(*p_left[1], *p_right[1])] = 1
    x_mask = x_mask.astype(bool)

    total_mask = y_mask & x_mask

    selected = np.zeros_like(area)
    selected[total_mask] = 1

    selected = ndimage.rotate(selected, -90)
    new_source_mask = source_mask.copy()
    new_source_mask[selected.astype(bool)] = part

    return new_source_mask


def parsing(source_path, reference_path, checkpoint="79999_iter.pth", to_dilate=False, to_fill=False, save=True):
    source_name = os.path.basename(source_path)
    reference_name = os.path.basename(reference_path)

    source_parent_dir = os.path.dirname(source_path)
    reference_parent_dir = os.path.dirname(reference_path)

    source_seg_dir = source_parent_dir + "/parsing"
    reference_seg_dir = reference_parent_dir + "/parsing"

    os.makedirs(source_seg_dir, exist_ok=True)
    os.makedirs(reference_seg_dir, exist_ok=True)

    source_seg = evaluate_image(source_path, cp=checkpoint)
    reference_seg = evaluate_image(reference_path, cp=checkpoint)

    new_source_seg_path = f"{source_seg_dir}/{source_name}"
    new_reference_seg_path = f"{reference_seg_dir}/{reference_name}"
    new_source_seg = handle_parsing(source_seg[1].astype(np.uint8))
    new_reference_seg = handle_parsing(reference_seg[1].astype(np.uint8))

    for p in (1, 6):
        if to_dilate:
            new_source_seg = dilate(new_source_seg, np.ones((5, 5)), iterations=3, part=p)
            new_reference_seg = dilate(new_reference_seg, np.ones((5, 5)), iterations=3, part=p)

        if to_fill:
            new_source_seg = fill_between_eye_eyebrow(new_source_seg, part=p)
            new_reference_seg = fill_between_eye_eyebrow(new_reference_seg, part=p)

    new_source_seg = Image.fromarray(new_source_seg)
    new_reference_seg = Image.fromarray(new_reference_seg)

    if save:
        new_source_seg.save(new_source_seg_path)
        new_reference_seg.save(new_reference_seg_path)

    return (new_source_seg_path, new_source_seg), (new_reference_seg_path, new_reference_seg)


def transfer(source_path, reference_path, opt, checkpoint="79999_iter.pth", save=True):
    source_seg, reference_seg = parsing(source_path, reference_path, checkpoint=checkpoint, save=save)

    # dataloader = SCDataLoader(opt)
    SCGan = create_model(opt, None)
    # SCGan.test()
    results = SCGan.transfer(source_path, reference_path, source_seg[0], reference_seg[0])

    fig, ax = plt.subplots(1, 3, figsize=(24, 36))

    ax[0].imshow(results[0])
    ax[0].set_title("Source")
    ax[1].imshow(results[1])
    ax[1].set_title("Reference")
    ax[2].imshow(results[2])
    ax[2].set_title("Transfer")

    for a in ax:
        a.axis("off")

    # fig, ax = plt.subplots(2, 3, figsize=(24, 36))

    # ax[0][0].imshow(results[0])
    # ax[0][0].set_title("Source")
    # ax[0][1].imshow(results[1])
    # ax[0][1].set_title("Reference")
    # ax[0][2].imshow(results[2])
    # ax[0][2].set_title("Transfer")

    # ax[1][0].imshow(results[1])
    # ax[1][0].set_title("Source")
    # ax[1][1].imshow(results[0])
    # ax[1][1].set_title("Reference")
    # ax[1][2].imshow(results[3])
    # ax[1][2].set_title("Removal")

    # for a in ax[0]:
    #     a.axis("off")

    # for a in ax[1]:
    #     a.axis("off")

    plt.show()

    return results


if __name__ == "__main__":
    source_seg, reference_seg = parsing(
        "dataset/non-makeup/00313.jpg", "dataset/makeup/b7fd5266d01609248414333cdf0735fae6cd34e7.png", save=False
    )

    source_mask = np.array(source_seg[1]).astype(np.uint8)

    plt.imshow(source_mask)
    plt.show()

    # eye = np.zeros_like(source_mask)
    # brow = np.zeros_like(source_mask)
    # skin = np.zeros_like(source_mask)

    # eye[source_mask == 1] = 1
    # brow[source_mask == 2] = 1
    # skin[source_mask == 4] = 1

    # eye_opening = cv2.dilate(eye, np.ones((5, 5)), iterations=3)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    # ax[0].imshow(eye)
    # ax[1].imshow(eye_opening)
    # plt.show()

    # contoured, _ = cv2.findContours(eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # print(contoured[0].shape)

    # contoured = cv2.drawContours(np.zeros_like(eye), contoured, -1, 1, 3)

    # fig, ax = plt.subplots()
    # ax.imshow(contoured)
    # plt.show()

    # fig, ax = plt.subplots(1, 3, figsize=(10, 20))
    # ax[0].imshow(eye)
    # ax[1].imshow(brow)
    # ax[2].imshow(skin)
    # plt.show()

    # eye_opening = cv2.morphologyEx(eye, cv2.MORPH_OPEN, np.ones((2, 2)))
    # brow_opening = cv2.morphologyEx(brow, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=4)

    # fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    # ax[0][0].imshow(eye)
    # ax[0][1].imshow(eye_opening)
    # ax[1][0].imshow(brow)
    # ax[1][1].imshow(brow_opening)
    # plt.show()

    # msk = eye_opening + brow_opening

    # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # ax[0].imshow(msk)
    # plt.show()

    # fig, ax = plt.subplots(1, 3, figsize=(20, 60))

    # ax[0].imshow(asd)
    # ax[1].imshow(opening)
    # ax[2].imshow(dilate)
    # plt.show()
