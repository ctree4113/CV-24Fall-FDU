#! /bin/sh
cd ../..

python eval.py \
--dataset 'chase' \
--data_root '/data/agent/Jetbrains/test/CV-24Fall-FDU/data/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--decoder_attention \
--ckpt_path /data/agent/Jetbrains/test/CV-24Fall-FDU/ckpt/decoder_attention_only/CHASEDB1 \
--output_path /data/agent/Jetbrains/test/CV-24Fall-FDU/eval_results/decoder_attention_only/CHASEDB1 \
--num_pred 5
