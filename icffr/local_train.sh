#!/bin/bash

#time_stamp=$(date +%F-%H-%M-%S)
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export TIME_STAMP=$(date +%F-%H-%M-%S)

export TASK_DIR="/data/sunyufei/tasks/icffr_balance_ir34_"${TIME_STAMP}
export PRINT_DIR=$TASK_DIR"/logs"
export LOG_DIR=$TASK_DIR"/tensorboard"
export CKPT_DIR=$TASK_DIR"/ckpt"


if [ ! -d $TASK_DIR ]; then
    mkdir $TASK_DIR
fi

if [ ! -d $PRINT_DIR ]; then
    mkdir $PRINT_DIR
fi

if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi

if [ ! -d $CKPT_DIR ]; then
    mkdir $CKPT_DIR
fi

nohup python -u -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_icffr.py > $PRINT_DIR/train.log 2>&1 &