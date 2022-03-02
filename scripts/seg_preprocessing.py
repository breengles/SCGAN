import os

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def dilate_eye(img, kernel, iterations=1, part=1):
    tmp = np.zeros_like(img)

    mask = img == part
    tmp[mask] = 1
    out = cv2.dilate(tmp, kernel, iterations=iterations)

    out = out.astype(bool)
    img[out] = part

    return img


def cut_eye_out(seg, init_seg, part=1):
    init_part = init_seg == part
    seg[init_part] = 0
    return seg


def main(path="../MT-Dataset/parsing/makeup"):
    for img_name in tqdm(os.listdir(path), desc="Seg preprocessing..."):
        img = Image.open(os.path.join(path, img_name))
        img = np.array(img)
        new_seg = img.copy()
        for p in (1, 6):
            new_seg = dilate_eye(new_seg, np.ones((10, 10)), iterations=3, part=p)

        for p in (1, 6):
            new_seg = cut_eye_out(new_seg, img, part=p)

        Image.fromarray(new_seg).save(os.path.join(path, img_name))


if __name__ == "__main__":
    for path in ("../MT-Dataset/parsing/makeup", "../MT-Dataset/parsing/non-makeup"):
        main(path)
