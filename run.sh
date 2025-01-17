#!/usr/bin/env bash

# ./train.py configs/histomatching/accurate_l2.yaml --seed 42 --local

#./train.py configs/finetuning/base.yaml --seed 42 --local
#./train.py configs/finetuning/eye.yaml --seed 42 --local
#./train.py configs/finetuning/g_delay=1000.yaml --seed 42 --local
#./train.py configs/finetuning/g_delay=5000.yaml --seed 42 --local
#./train.py configs/finetuning/g_delay=10000.yaml --seed 42 --local
#./train.py configs/finetuning/g_delay=100000.yaml --seed 42 --local


./train.py configs/finetuning/eye/eye=5.yaml --seed 42 --local
./train.py configs/finetuning/eye/eye=10.yaml --seed 42 --local
./train.py configs/finetuning/eye/eye=100.yaml --seed 42 --local



wandb sync wandb/offline-*
