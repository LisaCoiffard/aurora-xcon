#!/bin/bash

# Array of loss types
loss_types=("mse" "triplet")


for loss_type in "${loss_types[@]}"; do
    python mnist_exp/train_ae_mnist.py  seed=$RANDOM loss_type=$loss_type wandb.mode=offline
done