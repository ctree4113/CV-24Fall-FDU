#! /bin/sh
cd ../..

python train.py \
--dataset 'chase' \
--data_root '/data/workspace_yi/cv/CV-24Fall-FDU/data/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 8 \
--epochs 150 \
--lr 0.002 \
--lr-update 'poly' \
--use_glfi \
--save /data/workspace_yi/cv/CV-24Fall-FDU/ckpt/glfi_only/CHASEDB1 \
--folds 5

