#!/usr/bin/env bash


./sc.py --num_epochs 1000 --log_step 16 --snapshot_step 200 --phase train --img_size 128 --nThreads 8 --serial_batches --dataroot "datasets/non-dilated/daniil.128px.cut.crop.overfit"

