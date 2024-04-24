# Minimizing the Accumulated Trajectory Error to Improve Dataset Distillation

### [Paper](https://arxiv.org/abs/2211.11004)
<br>

![Teaser image](docs/result_example.png)

This repo contains code for training expert trajectories and distilling synthetic data from our Dataset Distillation by FTD paper (CVPR 2023). 

> [**Minimizing the Accumulated Trajectory Error to Improve Dataset Distillation**](https://arxiv.org/abs/2211.11004)<br>
> [Jiawei Du*](https://scholar.google.com/citations?user=WrJKEzEAAAAJ&hl=en), [Yidi Jiang*](https://scholar.google.com/citations?hl=en&user=le6gC58AAAAJ), [Vincent Y. F. Tan](https://vyftan.github.io/), [Joey tianyi Zhou](https://joeyzhouty.github.io/), [Haizhou Li](https://colips.org/~eleliha/)<br>
> CFAR A*STAR, NUS<br>
> CVPR 2023

The task of "Dataset Distillation" is to learn a small number of synthetic images such that a model trained on this set alone will have similar test performance as a model trained on the full real dataset.




## Accumulated Trajectory Error

![Teaser image](docs/illustrate.png)

State-of-the-art methods primarily rely on
learning the synthetic dataset by matching the gradients obtained
during training between the real and synthetic data.
However, these gradient-matching methods suffer from the
so-called accumulated trajectory error caused by the discrepancy
between the distillation and subsequent evaluation. To
mitigate the adverse impact of this accumulated trajectory
error, we propose a novel approach that encourages the optimization
algorithm to seek a flat trajectory.

<img src='docs/accumulate_loss.png' width=600>

The flat trajectory distillation (FTD) in purple line mitigates the so-called accumulated trajectory error than the baseline in blue line. 

<br>



### Getting Started

First, create the conda virtual enviroment

```bash
conda env create -f enviroment.yaml
```

You can then activate your  conda environment with
```bash
conda activate distillation
```

### Generating Expert Trajectories
Before doing any distillation, you'll need to generate some expert trajectories using ```.\buffer\buffer.py```

The following command will train 100 ConvNet models on CIFAR-100 with ZCA whitening for 50 epochs each:
```bash
python buffer.py --dataset=CIFAR100 --model=ConvNet --train_epochs=50 --num_experts=100 --zca --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset} --rho 0.01
```
There is an example in the ```.\buffer\run_buffer.sh```

the default data and buffer storage path are ```.\data``` and ```.\buffer_storage```


### Distillation by Matching Training Trajectories

CIFAR-100 IPC50 with **Distribution Matching**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distill_FTD_noema.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --batch_syn=1000 --max_start_epoch=40 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer/buffer/ --data_path=/data/ --Iteration=5000 --student_distmatch --student_factor 50 --num_eval 2
```

CIFAR100 IPC50 baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distill_FTD_noema.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --batch_syn=1000 --max_start_epoch=40 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer/buffer/ --data_path=/data/ --Iteration=5000 --student_factor 0 --num_eval 2
```

<img src='docs/results.png' width=800 >

Please find a full list of hyper-parameters below:
<img src='docs/parameters.png' width=600>


