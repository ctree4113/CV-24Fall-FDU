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
--decoder_attention \
--use_mrde \
--use_glfi \
--freq_weight 0.05 \
--topo_weight 0.05 \
--save /data/agent/workspace_yi/cv/CV-24Fall-FDU/ckpt/fully_improved/ISIC2018 \
--folds 5
