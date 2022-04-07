#!/usr/bin/env bash


# ./train.py configs/test_dilate.yaml --seed 42 --local
# ./train.py configs/test_rect.yaml --seed 42 --local


#for cfg in configs/overfit/*
#do
#    ./train.py "$cfg" --local --seed 42
#done
#
#for cfg in configs/overfit_1example/*
#do
#    ./train.py "$cfg" --local --seed 42
#done
#
#wandb sync wandb/offline-*


# ./train.py configs/rect/beauty.yaml --seed 42 --local
# ./train.py configs/rect/merged.yaml --seed 42 --local

# ./train.py configs/dilate/beauty.yaml --seed 42 --local
# ./train.py configs/dilate/merged.yaml --seed 42 --local

# for cfg in configs/histomatching/*
# do
#     ./train.py "$cfg" --seed 42 --local
# done

# ./train.py configs/histomatching/fast.yaml --seed 42 --local
./train.py configs/histomatching/fast_l1.yaml --seed 42 --local
./train.py configs/histomatching/fast_l1_hsv.yaml --seed 42 --local

./train.py configs/finetuning/merged.yaml --seed 42 --local
./train.py configs/finetuning/merged_dilate=1000.yaml --seed 42 --local

./train.py configs/dilate/new_merged.yaml --seed 42 --local



wandb sync wandb/offline-*
