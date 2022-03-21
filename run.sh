#!/usr/bin/env bash


for cfg in configs/overfit/idt_*
do
  ./train.py "$cfg" --local
done


./train.py configs/beauty_dilate.yaml --local
./train.py configs/my_dilate.yaml --local
./train.py configs/merged_dilate.yaml --local

wandb sync wandb/offline-*
