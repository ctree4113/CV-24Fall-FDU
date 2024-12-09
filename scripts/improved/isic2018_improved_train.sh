#! /bin/sh
cd ../..

python train.py \
--dataset 'isic' \
--data_root '/data/agent/Jetbrains/test/CV-24Fall-FDU/data/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 4 \
--epochs 100 \
--lr 0.0000001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--model_type 'improved' \
--decoder_attention \
--use_mrde \
--use_glfi \
--freq_weight 0.05 \
--topo_weight 0.05 \
--save /data/agent/Jetbrains/test/CV-24Fall-FDU/ckpt/improved/ISIC2018 \
--folds 5

