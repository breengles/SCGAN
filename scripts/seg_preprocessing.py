import os

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


def dilate_eye(img, kernel, iterations=1, part=1, thickness=None):
    init_img = img.copy()
    tmp = np.zeros_like(img)

    mask = img == part
    tmp[mask] = 1
    dilated = cv2.dilate(tmp, kernel, iterations=iterations)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(dilated)

    if thickness is None:
        thickness = int(1 + max(out.shape[0], out.shape[1]) / 100 * 15)  # 15% contour

    out = cv2.drawContours(out, contours, -1, (128, 128, 128), thickness)

    out = out.astype(bool)
    img[out] = part

    # rewrite boundaries
    for p in (0, 2, 7, 8, 12):
        mask = img == p
        img[init_img == p] = p

    return img


def cut_eye_out(seg, init_seg, part=1):
    init_part = init_seg == part
    seg[init_part] = 0
    return seg


def main(path="../datasets/MT-Dataset", kind="makeup"):
    path = os.path.join(path, "segments", kind)
    for img_name in tqdm(os.listdir(path), desc=f"Seg ({kind}) preprocessing..."):
        img = Image.open(os.path.join(path, img_name))
        img = np.array(img)
        new_seg = img.copy()
        for p in (1, 6):
            new_seg = dilate_eye(new_seg, np.ones((3, 3)), iterations=3, part=p)

        for p in (1, 6):
            new_seg = cut_eye_out(new_seg, img, part=p)

        img = Image.fromarray(new_seg)
        img.save(os.path.join(path, img_name))


if __name__ == "__main__":
    dataset_library = "../datasets"
    dataset_names = ("daniil.128px.cut.crop.overfit",)

    for dataset_name in dataset_names:
        path = os.path.join(dataset_library, dataset_name)
        for kind in ("makeup", "non-makeup"):
            main(path, kind)
