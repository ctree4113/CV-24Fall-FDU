#! /bin/sh
cd ../..

python eval.py \
--dataset 'chase' \
--data_root '/data/agent/workspace_yi/cv/CV-24Fall-FDU/data/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--decoder_attention \
--use_glfi \
--use_mrde \
--ckpt_path /data/agent/workspace_yi/cv/CV-24Fall-FDU/ckpt/fully_improved/CHASEDB1 \
--output_path /data/agent/workspace_yi/cv/CV-24Fall-FDU/eval_results/improved/CHASEDB1 \
--num_pred 5

