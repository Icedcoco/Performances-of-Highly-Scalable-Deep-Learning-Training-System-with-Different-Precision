# Performances-of-Highly-Scalable-Deep-Learning-Training-System-with-Different-Precision
Project of NYU Tandon 2022S ECE-9143.  
Evaluating the DL Trainging system performance using different precisions.  
We measured the system performence using three different precisions -- FP32, AMP and FP16, discuss the pros and cons and why.  
Using ResNet based on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). 
Maintained by [Jingxuan Wang](https://github.com/Jexxie) and [Yuchen Kou](https://github.com/Icedcoco). 


- [Content in this repository](#content-in-this-repository)
- [Environment](#environment)
- [Usage](#usage)
  * [Example on HPC](#example-on-hpc)
- [Code Structure](#code-structure)
- [Results and Observation](#results-and-observation)
- [Challenges we met](#challenges-we-met)




# Content in this repository
# Environment
# Usage
(how to execute the code)
Platform: NYU HPC
In sbatch file:

Select network: ResNet18(Default), ResNet34(--use_34), ResNet50(--use_50)  
Select precision: FP32(Default), AMP(--use_amp), FP16(--use_half)  
If using Distributed_Data_Parallel, use "python -m torch.distributed.launch --nproc_per_node N main.py", where N is the GPU amount, and add parameter "--use_ddp"  
Other parameters used in the project:  
  Hardware: 4 CPU + 1 GPU per process  
  Hyperparameters: --num_workers=2 --optimizer='sgd' --use_cuda --lr=0.005 --epoch=600 --batch_size=128  
  
## Example on HPC
# Code Structure
# Results and Observation

|  Precision   | Training Time  |  Model Size  |  Bandwidth
|  ----  | ----  | ----  | ----  |
| fp16  | 单元格 | ----  | ----  |
| TF32  | 单元格 | ----  | ----  |
| mp  | 单元格 | ----  | ----  |
| fp32  | 单元格 | ----  | ----  |

# Challenges we met
 - When we started this project, the lecture did not involve relevant knowledge, and the principle was not clear. We spent a lot of time looking at the NVIDIA manual,  related papers and blogs to learn about the possible performance and differences between different precisions. 
 - We planned to use multiple machines in parallel to calculate under different precisions situations, and we were not sure if we could accomplish this task. 
 - We are not sure if the experimental results will be consistent with our prediction。 
 - 





