#! /bin/sh
cd ../..

python train.py \
--dataset 'chase' \
--data_root '/data/agent/workspace_yi/cv/CV-24Fall-FDU/data/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 2 \
--epochs 150 \
--lr 0.0001 \
--lr-update 'poly' \
--folds 5 \
--save /data/agent/workspace_yi/cv/CV-24Fall-FDU/ckpt/baseline/CHASEDB1

