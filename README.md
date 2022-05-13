# Performances-of-Highly-Scalable-Deep-Learning-Training-System-with-Different-Precision
Project of NYU Tandon 2022S ECE-9143.
Evaluating the DL Trainging system performance using different precisions.
We measured the system performence using three different precisions -- FP32, AMP and FP16, discuss the pros and cons and why.

## Usage(NYU HPC):
In sbatch file:

Select network: ResNet18(Default), ResNet34(--use_34), ResNet50(--use_50)
Select precision: FP32(Default), AMP(--use_amp), FP16(--use_half)
If using Distributed_Data_Parallel, use "python -m torch.distributed.launch --nproc_per_node N main.py", where N is the GPU amount, and add parameter "--use_ddp"
Other parameters used in the project:
Hardware: 4 CPU + 1 GPU per process
Hyperparameters: --num_workers=2 --optimizer='sgd' --use_cuda --lr=0.005 --epoch=600 --batch_size=128
