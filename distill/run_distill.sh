#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python distill_FTD.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca \
#    --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --buffer_path=../buffer/buffer/ --data_path=/data/ --ema_decay=0.9995 --Iteration=5000 --student_distmatch --student_factor 50


#CUDA_VISIBLE_DEVICES=0 python distill_FTD.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --zca \
#    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer/buffer/ --data_path=../dataset/ --ema_decay=0.9995 --Iteration=5000 --student_distmatch --student_factor 50

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distill_FTD.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --batch_syn=1000 --max_start_epoch=40 --zca \
     --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer/buffer/ --data_path=/data/ --ema_decay=0.999 --Iteration=5000 --student_distmatch --student_factor 50
