#!/bin/bash 
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=amp_50_005_4c_2w_128_ddp4
#SBATCH --output=%x.out

module purge
module load python/intel/3.8.6
module load anaconda3/2020.07
module load cuda/11.3.1

eval "$(conda shell.bash hook)"
conda activate hpml

# Select network: ResNet18(Default), ResNet34(--use_34), ResNet50(--use_50)
# Select precision: FP32(Default), AMP(--use_amp), FP16(--use_half)
# If using Distributed_Data_Parallel, use "python -m torch.distributed.launch --nproc_per_node N main.py",
# where N is the GPU amount, and add parameter "--use_ddp"
# Other parameters used in the project:
# Hardware: 4 CPU + 1 GPU per process
# Hyperparameters: --num_workers=2 --optimizer='sgd' --use_cuda --lr=0.005 --epoch=600 --batch_size=128

python -m torch.distributed.launch --nproc_per_node 4 main.py --num_workers=2 --optimizer='sgd' --use_cuda --lr=0.005 --epoch=600 --batch_size=128 --use_50 --use_amp --use_ddp

conda deactivate
