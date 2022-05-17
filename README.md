# Performances-of-Highly-Scalable-Deep-Learning-Training-System-with-Different-Precision
Project of NYU Tandon 2022S ECE-9143.  
Evaluating the DL Trainging system performance using different precisions.  
We measured the system performence using three different precisions -- FP32, AMP and FP16, discuss the pros and cons and why.  
Using model ResNet and dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). 
Maintained by [Jingxuan Wang](https://github.com/Jexxie) and [Yuchen Kou](https://github.com/Icedcoco). 


# Content in this repository

- [Content in this repository](#content-in-this-repository)
- [Environment](#environment)
- [Usage](#usage)
  * [Example on HPC](#example-on-hpc)
- [Code Structure](#code-structure)
- [Results and Observation](#results-and-observation)
- [Challenges we met](#challenges-we-met)



# Environment
### Platform:
NYU HPC  
### Hardware:
• Node type: Lenovo SR670  
• CPU: Intel Xeon Platinum 8268 24C 205W 2.9GHz  
• GPU: V100 (32 GB) NVIDIA GPU cards [3]  
### Software:
• OS: Red Hat Enterprise Linux release 8.4  
• Programming language: Python 3.8  
• Deep learning frameworks: PyTorch  

# Usage
Upload main.py file and .sbatch file in the same path, and run sbatch. 

## Example on HPC
In sbatch file:

Select network: ResNet18(Default), ResNet34(--use_34), ResNet50(--use_50)  
Select precision: FP32(Default), AMP(--use_amp), FP16(--use_half)  
If using Distributed_Data_Parallel, use "python -m torch.distributed.launch --nproc_per_node N main.py", where N is the GPU amount, and add parameter "--use_ddp"  
Other parameters used in the project:  
  Hardware: 4 CPU + 1 GPU per process  
  Hyperparameters: --num_workers=2 --optimizer='sgd' --use_cuda --lr=0.005 --epoch=600 --batch_size=128  
  
# Code Structure
# Results and Observation

| model_size | ResNet18 | ResNet50 |
| ----------------- | ----------- | ----------- |
| fp32  | 42.662MB | 89.928MB  |
| fp16  | 21.368MB | 45.167MB |
| amp  | 42.662MB | 89.928MB |
| Compression ratio  | 0.500867282  | 0.502257361  |

Compression ratio are both close to 0.5. 

| batchsize-machine | precision | time        | train       | compute     | data        | move        | test   |
| ----------------- | --------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------ |
| 128-SG            | fp32      | 0.039667519 | 0.036163683 | 0.033120205 | 0.002148338 | 0.000521739 | 0.0145 |
| 128-SG            | fp16      | 0.022506394 | 0.019053708 | 0.015780051 | 0.002250639 | 0.000595908 | 0.0132 |
| 128-SG            | amp       | 0.027570332 | 0.022506394 | 0.019002558 | 0.002506394 | 0.000613811 | 0.0158 |
| 64-DDP            | fp32      | 0.03255102  | 0.029081633 | 0.025663265 | 0.00255102  | 0.000357143 | 0.0276 |
| 64-DDP            | fp16      | 0.023622449 | 0.020510204 | 0.016887755 | 0.00255102  | 0.000377551 | 0.0252 |
| 64-DDP            | amp       | 0.027091837 | 0.02377551  | 0.020204082 | 0.00255102  | 0.000357143 | 0.0276 |
| 128-DDP           | fp32      | 0.056734694 | 0.049081633 | 0.041428571 | 0.005714286 | 0.000826531 | 0.0308 |
| 128-DDP           | fp16      | 0.033469388 | 0.02622449  | 0.018469388 | 0.005714286 | 0.000826531 | 0.0288 |
| 128-DDP           | amp       | 0.036020408 | 0.028163265 | 0.020612245 | 0.005612245 | 0.000795918 | 0.0272 |


# Challenges we met
 - When we started this project, the lecture did not involve relevant knowledge, and the principle was not clear. We spent a lot of time looking at the NVIDIA manual,  related papers and blogs to learn about the possible performance and differences between different precisions. 
 - We planned to use multiple machines in parallel to calculate under different precisions situations, and we were not sure if we could accomplish this task. 
 - We are not sure if the experimental results will be consistent with our prediction. 
 - Due to the unavailability of NVIDIA A100 resources, we cancelled our plans to include TF32 (tensorflow float 32) in the project.




