#! /bin/sh
cd ../..

python train.py \
--dataset 'chase' \
--data_root '/data/agent/Jetbrains/test/CV-24Fall-FDU/data/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 4 \
--epochs 150 \
--lr 0.002 \
--lr-update 'poly' \
--use_mrde \
--save /data/agent/Jetbrains/test/CV-24Fall-FDU/ckpt/mrde_only/CHASEDB1 \
--folds 5

