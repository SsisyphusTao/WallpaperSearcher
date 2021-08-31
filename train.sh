#! /bin/bash
time \
OMP_NUM_THREADS=24 \
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
            --nproc_per_node=2 train.py \
            > local.log