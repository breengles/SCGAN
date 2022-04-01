#!/usr/bin/env bash


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



./train.py configs/default_beauty.yaml --local --seed 42
./train.py configs/default_daniil.yaml --local --seed 42
./train.py configs/default_merged.yaml --local --seed 42
./train.py configs/default_my.yaml --local --seed 42




wandb sync wandb/offline-*
