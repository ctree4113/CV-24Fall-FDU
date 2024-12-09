#! /bin/sh
cd ..
# python eval.py \
# --dataset 'isic' \
# --data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
# --resize 224 320 \
# --metrics DSC:IOU:ACC:PREC \
# --num-class 1 \
# --decoder_attention \
# --ckpt_path /root/autodl-tmp/ckpt/decoder_attention/ISIC2018 \
# --output_path /root/autodl-tmp/eval_results/decoder_attention/ISIC2018 \
# --num_pred 5

python eval.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--metrics DSC:IOU:ACC:PREC \
--num-class 1 \
--ckpt_path /root/autodl-tmp/ckpt/baseline/ISIC2018 \
--output_path /root/autodl-tmp/eval_results/baseline/ISIC2018 \
--num_pred 5



