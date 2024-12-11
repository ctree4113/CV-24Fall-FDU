#! /bin/sh
cd ../..

python train.py \
--dataset 'chase' \
--data_root '/data/agent/Jetbrains/test/CV-24Fall-FDU/data/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 2 \
--epochs 150 \
--lr 0.002 \
--lr-update 'poly' \
--freq_weight 0.05 \
--topo_weight 0.05 \
--save /data/agent/Jetbrains/test/CV-24Fall-FDU/ckpt/improved_loss_only/CHASEDB1 \
--folds 5

