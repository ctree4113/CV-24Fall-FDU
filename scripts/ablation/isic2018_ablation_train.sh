#! /bin/sh
cd ../..

# Baseline
python train.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 10 \
--epochs 100 \
--lr 0.0001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--model_type 'base' \
--save_dir '/root/autodl-tmp/ablation/ISIC2018/baseline' \
--folds 5

# MRDE Only
python train.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 10 \
--epochs 100 \
--lr 0.0001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--model_type 'improved' \
--use_mrde \
--save_dir '/root/autodl-tmp/ablation/ISIC2018/mrde_only' \
--folds 5

# GLFI Only
python train.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 10 \
--epochs 100 \
--lr 0.0001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--model_type 'improved' \
--use_glfi \
--save_dir '/root/autodl-tmp/ablation/ISIC2018/glfi_only' \
--folds 5

# Full Model
python train.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 10 \
--epochs 100 \
--lr 0.0001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--model_type 'improved' \
--use_mrde \
--use_glfi \
--freq_weight 0.1 \
--topo_weight 0.1 \
--save_dir '/root/autodl-tmp/ablation/ISIC2018/full_model' \
--folds 5 