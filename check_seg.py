#!/usr/bin/env python

from PIL import Image
from matplotlib import pyplot as plt
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path")

    args = parser.parse_args()

    img = Image.open(args.path)
    plt.imshow(img)
    plt.show()
