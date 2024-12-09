#! /bin/sh
cd ../..

python train.py \
--dataset 'isic' \
--data_root '/data/agent/Jetbrains/test/CV-24Fall-FDU/data/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 4 \
--epochs 100 \
--lr 0.0001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--model_type 'improved' \
--use_glfi \
--save /data/agent/Jetbrains/test/CV-24Fall-FDU/ckpt/glfi_only/ISIC2018 \
--folds 5

