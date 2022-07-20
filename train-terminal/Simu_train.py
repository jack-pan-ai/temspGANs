#!/usr/bin/env bash

import os

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_GAN.py \
--gpu 0 \
--latent_dim 128 \
--simu_dim 64 \
--simu_channels 64 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 1 \
--n_gen 1 \
--patch_size 4 \
--batch_size 128 \
--epochs 1500 \
--eval_epochs 5 \
--tau 1.0 \
--nu 0.5 \
--rho 1.0 \
--print_freq 50 \
--exp_name GRF-2D-v1.0")