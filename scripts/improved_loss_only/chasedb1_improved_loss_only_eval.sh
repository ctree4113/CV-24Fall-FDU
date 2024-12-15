#! /bin/sh
cd ../..

python eval.py \
--dataset 'chase' \
--data_root '/data/agent/workspace_yi/cv/CV-24Fall-FDU/data/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--ckpt_path /data/agent/workspace_yi/cv/CV-24Fall-FDU/ckpt/improved_loss_only/CHASEDB1 \
--output_path /data/agent/workspace_yi/cv/CV-24Fall-FDU/eval_results/improved_loss_only/CHASEDB1 \
--num_pred 5

