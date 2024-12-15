#! /bin/sh
cd ../..

python train.py \
--dataset 'isic' \
--data_root '/data/agent/workspace_yi/cv/CV-24Fall-FDU/data/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 8 \
--epochs 100 \
--lr 0.0001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--freq_weight 0.05 \
--topo_weight 0.05 \
--save /data/agent/workspace_yi/cv/CV-24Fall-FDU/ckpt/improved_loss_only/ISIC2018 \
--folds 5

