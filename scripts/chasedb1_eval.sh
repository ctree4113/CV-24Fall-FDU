#! /bin/sh
cd ..
# python eval.py \
# --dataset 'chase' \
# --data_root '/root/autodl-tmp/datas/CHASEDB1' \
# --resize 960 960 \
# --metrics clDice:DSC:IOU:1-Betti \
# --num-class 1 \
# --decoder_attention \
# --ckpt_path /root/autodl-tmp/ckpt/decoder_attention/CHASEDB1 \
# --output_path /root/autodl-tmp/eval_results/decoder_attention/CHASEDB1 \
# --num_pred 5

python eval.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--ckpt_path /root/autodl-tmp/ckpt/baseline/CHASEDB1 \
--output_path /root/autodl-tmp/eval_results/baseline/CHASEDB1 \
--num_pred 5


