#! /usr/bin/env bash


for cfg in configs/overfit/single/*
do
    ./train.py $cfg --local
done

wandb sync wandb/offline-*



./train.py configs/default_beauty.yaml --local
./train.py configs/default_daniil.yaml --local
./train.py configs/default_merged.yaml --local
./train.py configs/default_my.yaml --local





wandb sync wandb/offline-*
