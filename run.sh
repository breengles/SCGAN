#!/usr/bin/env bash


for cfg in configs/idt_hypertuning/dilate/*
do
  ./train.py "$cfg" --local
done


for cfg in configs/idt_hypertuning/rect/*
do
  ./train.py "$cfg" --local
done



# ./train.py configs/beauty_dilate.yaml --local
# ./train.py configs/my_dilate.yaml --local
# ./train.py configs/merged_dilate.yaml --local

wandb sync wandb/offline-*
