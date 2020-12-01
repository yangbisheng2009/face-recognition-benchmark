#!/bin/bash

if [ "$1" = "train" ];
then
    #nohup python train.py --train-path /home/work/datasets/face-reco/train-data/ --backbone mobile --batch-size 512 > log.log 2> err.log &
    nohup python train.py --train-path /home/work/datasets/face-reco/train-data/ --backbone resnet101 --batch-size 64 > log.log 2> err.log &
elif [ "$1" = "predict" ];
then
    echo "predict"
fi
