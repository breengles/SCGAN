#!/usr/bin/env bash


./train.py configs/overfit/dilate/default_dilate_1example.yaml --local
./train.py configs/overfit/rect/default_rect_1example.yaml --local


./train.py configs/dilate/beauty.yaml --local
./train.py configs/dilate/my.yaml --local
./train.py configs/dilate/merged.yaml --local

./train.py configs/rect/beauty.yaml --local
./train.py configs/rect/my.yaml --local
./train.py configs/rect/merged.yaml --local


wandb sync wandb/offline-*
