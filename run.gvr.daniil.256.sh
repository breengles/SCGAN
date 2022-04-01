#!/bin/bash

#SBATCH -J scgan.daniil.256
#SBATCH -p dgx
#SBATCH -w dgx02
#SBATCH --reservation=mldl
#SBATCH --gres=gpu:1

#SBATCH -t 03-00:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBTACH --mem 32G

#SBATCH --no-requeue
#SBATCH --no-kill

#SBATCH -o log_slurm_job.%j.%N.std_out_err


module add Python/v3.6.5
# module add cuda/v10.1-1

echo -e "\n\n loaded modules:"
module list


echo -e "\nMy PATH=$PATH\n"
echo -e "Python -v :\c"; python -V

echo -e "Pip -v :\c"; pip -V
echo -e "Pip packages:\n"
pip list

#echo -e "Conda envs:\n"
#conda env list

#echo -e "Conda packages:\n"
#conda list

echo -e "Nvidia:\n"
nvidia-smi

export WANDB_MODE="offline"


python train.py configs/gvr/daniil.256.yaml --seed 42 --local
