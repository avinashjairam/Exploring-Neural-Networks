#!/bin/bash


# adam optimizer
CUDA_VISIBLE_DEVICES=0 python train.py \
     --ckpt-dir models/adam_optimizer_dropout0.3 \
     --optimizer-name adam | tee logs/adam_dropout0.3.txt

# SGD with nesterov optimizer
CUDA_VISIBLE_DEVICES=0 python train.py \
     --ckpt-dir models/SGD_with_nesterov_optimizer_dropout0.3 \
     --optimizer-name SGD_with_nesterov | tee logs/SGD_with_nesterov_dropout0.3.txt

# batchnorm
CUDA_VISIBLE_DEVICES=0 python train.py \
     --ckpt-dir models/adam_batchnorm_dropout0.3 \
     --optimizer-name adam \
     --use-batch-norm | tee logs/adam_batchnorm_dropout0.3.txt

# add data augmentation
CUDA_VISIBLE_DEVICES=0 python train.py \
     --ckpt-dir models/adam_batchnorm_dataaug_dropout0.3 \
     --optimizer-name adam \
     --use-batch-norm \
     --use-data-augmentation | tee logs/adam_batchnorm_dataaug_dropout0.3.txt

# add xavier initialization
CUDA_VISIBLE_DEVICES=0 python train.py \
     --ckpt-dir models/adam_batchnorm_dataaug_xavier_dropout0.3 \
     --optimizer-name adam \
     --use-batch-norm \
     --use-data-augmentation \
     --use-xavier-init | tee logs/adam_batchnorm_dataaug_xavier_dropout0.3.txt


# tune dropout
for p in "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
         --ckpt-dir models/adam_batchnorm_dataaug_xavier_dropout$p \
         --optimizer-name adam \
         --use-batch-norm \
         --use-data-augmentation \
         --use-xavier-init \
         --dropout-prob $p | tee logs/adam_batchnorm_dataaug_xavier_dropout$p.txt
done


# add learning rate scheduler
CUDA_VISIBLE_DEVICES=0 python train.py \
     --ckpt-dir models/adam_batchnorm_dataaug_xavier_lrscheduler_dropout0.2 \
     --optimizer-name adam \
     --use-batch-norm \
     --use-data-augmentation \
     --use-xavier-init \
     --use-lr-scheduler \
     --dropout-prob 0.2 | tee logs/adam_batchnorm_dataaug_xavier_lrscheduler_dropout0.2.txt


for seed in "98" "99" "100" "101" "102"
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
         --ckpt-dir models/adam_optimizer_dropout0.3_seed$seed \
         --optimizer-name adam \
         --seed $seed | tee logs/adam_dropout0.3_seed$seed.txt
done