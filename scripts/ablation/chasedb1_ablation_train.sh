#! /bin/sh
cd ../..

# Baseline
python train.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 4 \
--epochs 150 \
--lr 0.002 \
--lr-update 'poly' \
--model_type 'base' \
--save_dir '/root/autodl-tmp/ablation/CHASEDB1/baseline' \
--folds 5

# MRDE Only
python train.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 4 \
--epochs 150 \
--lr 0.002 \
--lr-update 'poly' \
--model_type 'improved' \
--use_mrde \
--save_dir '/root/autodl-tmp/ablation/CHASEDB1/mrde_only' \
--folds 5

# GLFI Only
python train.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 4 \
--epochs 150 \
--lr 0.002 \
--lr-update 'poly' \
--model_type 'improved' \
--use_glfi \
--save_dir '/root/autodl-tmp/ablation/CHASEDB1/glfi_only' \
--folds 5

# Full Model
python train.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 4 \
--epochs 150 \
--lr 0.002 \
--lr-update 'poly' \
--model_type 'improved' \
--use_mrde \
--use_glfi \
--freq_weight 0.1 \
--topo_weight 0.1 \
--save_dir '/root/autodl-tmp/ablation/CHASEDB1/full_model' \
--folds 5 